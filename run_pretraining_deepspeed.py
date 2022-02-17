"""BERT pretraining runner with DeepSpeed"""

import argparse
import functools
import json
import loggerplus as logger
import math
import numpy as np
import os
import random
import torch

from time import perf_counter

from torch.utils.data import DataLoader
from torch import profiler
from tqdm import tqdm
from pathlib import Path

import deepspeed
import kfac
from apex.optimizers import FusedLAMB

import src.modeling_deepspeed as modeling
from src.dataset import ShardedPretrainingDataset, DistributedSampler
from src.schedulers import PolyWarmUpScheduler, LinearWarmUpScheduler
from src.tokenization import get_wordpiece_tokenizer, get_bpe_tokenizer
from src.utils import is_main_process, get_world_size, get_rank


class BertPretrainingCriterion(torch.nn.Module):
    def __init__(self, vocab_size):
        super(BertPretrainingCriterion, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size

    def forward(self, prediction_scores, masked_lm_labels,
            seq_relationship_score=None, next_sentence_labels=None):
        masked_lm_loss = self.loss_fn(prediction_scores.view(-1, self.vocab_size),
                                      masked_lm_labels.view(-1))
        if seq_relationship_score is not None and next_sentence_labels is not None:
            next_sentence_loss = self.loss_fn(seq_relationship_score.view(-1, 2),
                                              next_sentence_labels.view(-1))
            return masked_lm_loss + next_sentence_loss
        return masked_lm_loss


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", required=True,
                        help="JSON config file")
    parser.add_argument("--model_config_file", required=True,
                        help="Model config file")
    # Set by torch.distributed.launch
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training')

    args = parser.parse_args()

    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])

    with open(args.config_file) as cfg:
        config = json.load(cfg)

    config['local_rank'] = args.local_rank
    config['checkpoint_dir'] = os.path.join(config['output_dir'], 'pretrain_ckpts')

    with open(args.model_config_file) as cfg:
        model_config = json.load(cfg)
        # Add full config to model config so we can pass deepspeed configuration
        # to BertEncoder to use custom deepspeed transformer kernels
        model_config['full_config'] = config

    return config, model_config


def setup_training(output_dir, checkpoint_dir, log_prefix):
    assert (torch.cuda.is_available())

    deepspeed.init_distributed(dist_backend='nccl', init_method='env://')

    if is_main_process():
        os.makedirs(checkpoint_dir, exist_ok=True)

    logger.init(
        handlers=[
            logger.StreamHandler(verbose=is_main_process()),
            logger.FileHandler(
                    os.path.join(output_dir, log_prefix + '.txt'),
                    overwrite=False, verbose=is_main_process()),
            logger.TorchTensorboardHandler(
                    os.path.join(output_dir, 'tensorboard'),
                    verbose=is_main_process()),
            logger.CSVHandler(
                    os.path.join(output_dir, log_prefix + '_metrics.csv'),
                    overwrite=False, verbose=is_main_process()),
        ]
    )

    logger.info('Torch distributed initialized (world_size={}, backend={})'.format(
            get_world_size(), torch.distributed.get_backend()))


def init_model(model_config):
    config = modeling.BertConfig.from_dict(model_config)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
    model = modeling.BertForPreTraining(config)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm',
                'LayerNorm.bias', 'LayerNorm.weight']
    if config.full_config['deepspeed']['transformer_kernel']:
        no_decay = no_decay + [
            'attn_nw', 'attn_nb', 'norm_w', 'norm_b', 'attn_qkvb', 'attn_ob',
            'inter_b', 'output_b'
        ]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    criterion = BertPretrainingCriterion(config.vocab_size)

    return model, optimizer_grouped_parameters, criterion


def load_checkpoint(model, ckpt_dir, ckpt_id=None):
    load_path, state_dict = model.load_checkpoint(
        ckpt_dir, ckpt_id, load_optimizer_states=False
    )
    if load_path is None:
        if ckpt_id is None:
            logger.info(f'Failed to load latest checkpoint from {ckpt_dir}. '
                        'Starting new training.')
        else:
            logger.info(f'Failed to load step {ckpt_id} checkpoint from '
                        f'{ckpt_dir}. Starting new training.')
        return 0, 0, None
    
    epoch = state_dict['epoch']
    global_step = state_dict['global_step']
    logger.info(f'Loaded model checkpoint {load_path}')
    return epoch, global_step, state_dict


def save_checkpoint(
    model, sampler, epoch, global_step, ckpt_dir, scheduler=None, preconditioner=None
):
    state_dict = {
        'epoch': epoch,
        'global_step': global_step,
        'sampler': sampler.state_dict(),
    }
    if preconditioner is not None:
        state_dict['preconditioner'] = preconditioner.state_dict()
    if scheduler is not None:
        state_dict['scheduler'] = scheduler.state_dict()
    success = model.save_checkpoint(ckpt_dir, global_step, state_dict)
    if success:
        logger.info(f'Saved checkpoint for step {global_step} to {ckpt_dir}')
    else:
        logger.info(f'Failed to save checkpoint for step {global_step} to '
                    f'{ckpt_dir}')


def prepare_dataset(config, model_config, model, sampler_state_dict=None):
    input_files = []
    if os.path.isfile(config['input_dir']):
        input_files.append(config['input_dir'])
    elif os.path.isdir(config['input_dir']):
        for path in Path(config['input_dir']).rglob('*.hdf5'):
            if path.is_file():
                input_files.append(str(path))

    vocab_size = model_config['vocab_size']
    vocab_file = model_config['vocab_file']
    lowercase = model_config['lowercase']
    tokenizer = model_config['tokenizer']

    mask_token_id = None
    if tokenizer == 'wordpiece':
        tokenizer = get_wordpiece_tokenizer(vocab_file, uppercase=not lowercase)
    elif tokenizer == 'bpe':
        tokenizer = get_bpe_tokenizer(vocab_file, uppercase=not lowercase)
    else:
        raise ValueError('Unknown tokenizer \'{}\'. Options are '
                         '\'wordpiece\' and \'bpe\''.format(tokenizer))
    mask_token_id = tokenizer.token_to_id('[MASK]')

    dataset = ShardedPretrainingDataset(
        input_files,
        mask_token_id, 
        config['dataset']['max_predictions_per_seq'],
        config['dataset']['masked_token_fraction'],
        vocab_size=vocab_size
    )
    sampler = DistributedSampler(dataset, get_world_size(), rank=get_rank())

    # TODO(gpauloski): fix bug with loading state (maybe its the prefetch
    # factor being two high so an index being prefetched is not in the
    # current file?)
    if sampler_state_dict is not None:
        sampler.load_state_dict(sampler_state_dict)

    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=model.train_micro_batch_size_per_gpu(),
        num_workers=4,
        pin_memory=True,
        #prefetch_factor=4,
        #persistent_workers=True
    )

    if is_main_process():
        logger.info('Samples in dataset: {}'.format(len(dataset)))
        logger.info('Samples per device: {}'.format(len(sampler)))
        logger.info('Sampler starting index: {}'.format(sampler.index))
        logger.info('Batches in dataloader: {}'.format(len(loader)))
    return loader, sampler


def main(config, model_config):
    model, optimizer_grouped_parameters, criterion = init_model(model_config)

    # Ensure train_batch_size = micro_batch_size * accum_steps * world_size
    ds_config = config['deepspeed']
    if 'gradient_accumulation' not in ds_config:
        ds_config['gradient_accumulation'] = int(math.ceil(
            ds_config['train_batch_size'] / (
            ds_config['train_micro_batch_size_per_gpu'] * get_world_size())
        ))
    effective_batch_size = (
        ds_config['train_micro_batch_size_per_gpu'] *
        ds_config['gradient_accumulation'] * get_world_size()
    )
    if effective_batch_size != ds_config['train_batch_size']:
        ds_config['train_batch_size'] = effective_batch_size

    optimizer_ = FusedLAMB(
        optimizer_grouped_parameters,
        lr=config['lr']
    )
    model, optimizer, _, scheduler = deepspeed.initialize(
        config=config['deepspeed'],
        optimizer=optimizer_,
        model=model,
        model_parameters=optimizer_grouped_parameters,
    )
    
    if config['lr_decay'] == 'poly':
        Scheduler = PolyWarmUpScheduler
    elif config['lr_decay'] == 'linear':
        Scheduler = LinearWarmUpScheduler
    else:
        raise ValueError('Unknown lr decay "{}"'.format(config['lr_decay']))

    schedulers = [Scheduler(
        optimizer_,
        warmup=config['warmup_proportion'],
        total_steps=config['max_steps'] - config['scheduler_offset_steps']
    )]

    if 'allreduce_bucket_cap_mb' in config:
        allreduce_bucket_cap_mb = config['allreduce_bucket_cap_mb']
    else:
        allreduce_bucket_cap_mb = 25

    if 'kfac' in config and config['kfac']:
        preconditioner = kfac.KFAC(
            model,
            factor_update_steps=config['kfac_factor_update_steps'],
            inv_update_steps=config['kfac_inv_update_steps'],
            lr=config['lr'],
            damping=0.003,
            factor_decay=0.95,
            kl_clip=0.001,
            accumulation_steps=model.gradient_accumulation_steps(),
            allreduce_bucket_cap_mb=allreduce_bucket_cap_mb,
            colocate_factors=True,
            compute_eigenvalue_outer_product=True,
            grad_worker_fraction=kfac.DistributedStrategy.COMM_OPT,
            grad_scaler=None,
            skip_layers=['BertLMPredictionHead', 'embedding'],
            verbose=True
        )
        if is_main_process():
            logger.info(preconditioner)
        schedulers.append(PolyWarmUpScheduler(
            preconditioner,
            warmup=config['warmup_proportion'],
            total_steps=config['max_steps'] - config['scheduler_offset_steps']
        ))
    else:
        preconditioner = None
    epoch, global_step, state_dict = load_checkpoint(
        model, config['checkpoint_dir']
    )

    if (
        state_dict is not None and 'optimizer' in state_dict and 
        # Do not restore optimizer state if new phase
        global_step > config['scheduler_offset_steps']
    ):
        model.optimizer.load_state_dict(state_dict['optimizer'])

    if (
        state_dict is not None and 'scheduler' in state_dict and
        # Skip loading scheduler state dict if starting new phase
        global_step != config['scheduler_offset_steps']
    ):
        for scheduler in schedulers:
            scheduler.load_state_dict(state_dict['scheduler'])

    if (
        preconditioner is not None and 
        state_dict is not None and
        'preconditioner' in state_dict
    ):
        preconditioner.load_state_dict(state_dict['preconditioner'])

    dataloader, sampler = prepare_dataset(
        config,
        model_config,
        model,
        sampler_state_dict=None if state_dict is None else state_dict['sampler']
    )
 
    if not config['disable_progress_bar']:
        train_iter = tqdm(dataloader, disable=not is_main_process())
    else:
        train_iter = dataloader

    profile = 'profile' in config and config['profile']
    if profile:
        logger.info('Enabling profiling')
        prof = profiler.profile(
            schedule=profiler.schedule(
                wait=0, warmup=2, active=3, repeat=0, skip_first=2,
            ),
            on_trace_ready=profiler.tensorboard_trace_handler(
                os.path.join(config['output_dir'], 'tensorboard')
            ),
        )

    model.train()
    start_time = perf_counter()
    step = 0
    avg_loss = 0
    samples = 0
    train_perf_time = perf_counter()

    if profile:
        prof.start()
    while True:
        for batch in train_iter:
            # Forward/Backward
            batch = [t.to(model.device) for t in batch]
            input_ids, segment_ids, input_mask, masked_labels, ns_labels = batch
            prediction_scores, seq_relationship_score = model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask
            )
            loss = criterion(prediction_scores, masked_labels,
                             seq_relationship_score, ns_labels)
            unscaled_loss = loss.item()
            avg_loss += unscaled_loss
            model.backward(loss)
            samples += get_world_size() * model.train_micro_batch_size_per_gpu()

            # Optimization step/logging/scheduler update
            if model.is_gradient_accumulation_boundary():
                for scheduler in schedulers:
                    scheduler.step()
                if preconditioner is not None:
                    preconditioner.step()
                model.step()

                avg_loss = avg_loss / model.gradient_accumulation_steps()
                global_step += 1
                step += 1

                if global_step % config['log_steps'] == 0:
                    logger.log(
                        tag='train',
                        step=global_step,  
                        epoch=epoch,
                        average_loss=avg_loss,
                        step_loss=unscaled_loss,
                        learning_rate=optimizer.param_groups[0]['lr'],
                        samples_per_second=
                            samples / (perf_counter() - train_perf_time)
                    )

                if (
                    global_step % config['checkpoint_steps'] == 0 or
                    global_step == config['max_steps'] or 
                    step == config['steps']
                ):
                    save_checkpoint(
                        model, sampler, epoch, global_step,
                        config['checkpoint_dir'], schedulers[0], preconditioner
                    )

                if (
                    global_step >= config['max_steps'] or 
                    step >= config['steps']
                ):
                    if profile:
                        prof.stop()
                    return global_step, perf_counter() - start_time

                avg_loss = 0
                if profile:
                    prof.step()
            else:
                model.step()

        epoch += 1

    if profile:
        prof.stop()


if __name__ == "__main__":
    config, model_config = parse_arguments()
    
    random.seed(config['seed'] + config['local_rank'])
    np.random.seed(config['seed'] + config['local_rank'])
    torch.manual_seed(config['seed'] + config['local_rank'])
    torch.cuda.manual_seed(config['seed'] + config['local_rank'])

    args = setup_training(
        config['output_dir'],
        config['checkpoint_dir'],
        config['log_prefix']
    )
   
    logger.info('TRAINING CONFIG:\n{}'.format(
                json.dumps(config, indent=2, sort_keys=True)))
    logger.info('MODEL CONFIG:\n{}'.format(
                json.dumps(model_config, indent=2, sort_keys=True)))

    start_time = perf_counter()
    global_step, train_time = main(config, model_config)
    runtime = perf_counter() - start_time

    logger.info("runtime: {}s  train_time: {}s".format(
                runtime, train_time))
