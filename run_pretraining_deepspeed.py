"""BERT pretraining runner with DeepSpeed"""

import argparse
import json
import loggerplus as logger
import math
import numpy as np
import os
import random
import torch

from time import perf_counter

from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

import deepspeed

import src.deepspeed.modeling as modeling
from src.dataset import ShardedPretrainingDataset, DistributedSampler
from src.schedulers import warmup_exp_decay_exp
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

    with open(args.config_file) as cfg:
        config = json.load(cfg)

    config['local_rank'] = args.local_rank
    config['checkpoint_dir'] = os.path.join(config['output_dir'], 'saved_models')

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
    load_path, state_dict = model.load_checkpoint(ckpt_dir, ckpt_id)
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
    sampler_state_dict = state_dict['sampler']
    logger.info(f'Loaded checkpoint {load_path}')
    return epoch, global_step, sampler_state_dict


def save_checkpoint(model, sampler, epoch, global_step, ckpt_dir):
    state_dict = {
        'epoch': epoch,
        'global_step': global_step,
        'sampler': sampler.state_dict(),
    }
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

    if sampler_state_dict is not None:
        sampler.load_state_dict(sampler_state_dict)

    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=model.train_micro_batch_size_per_gpu(),
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    if is_main_process():
        logger.info('Samples in dataset: {}'.format(len(dataset)))
        logger.info('Samples per device: {}'.format(len(sampler)))
        logger.info('Sampler starting index: {}'.format(sampler.index))
        logger.info('Batches in dataloader: {}'.format(len(loader)))
    return loader, sampler


def update_lr(optimizer, optimizer_steps, config):
    lr = config['lr']
    lr *= warmup_exp_decay_exp(
        optimizer_steps,
        config['decay_rate'],
        config['decay_steps'],
        config['max_steps'],
        config['warmup_proportion']
    )
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main(config, model_config):
    model, optimizer_grouped_parameters, criterion = init_model(model_config)
    model, optimizer, _, _ = deepspeed.initialize(
        config=config['deepspeed'],
        model=model,
        model_parameters=optimizer_grouped_parameters
    )
    epoch, global_step, sampler_state_dict = load_checkpoint(
        model, config['checkpoint_dir']
    )
    dataloader, sampler = prepare_dataset(
        config,
        model_config,
        model,
        sampler_state_dict=sampler_state_dict
    )
 
    if not config['disable_progress_bar']:
        train_iter = tqdm(dataloader, disable=not is_main_process())
    else:
        train_iter = dataloader

    model.train()
    start_time = perf_counter()
    step = 0
    avg_loss = 0

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

            # Optimization step/logging/scheduler update
            if model.is_gradient_accumulation_boundary():
                if model.fp16_enabled():
                    lr = update_lr(
                        optimizer,
                        global_step - config['scheduler_offset_steps'],
                        config
                    )
                model.step()

                avg_loss = avg_loss / model.gradient_accumulation_steps()
                if global_step % config['log_steps'] == 0:
                    logger.log(
                        tag='train',
                        step=global_step,
                        epoch=epoch,
                        average_loss=avg_loss,
                        step_loss=unscaled_loss,
                        learning_rate=lr
                    )

                if (
                    global_step % config['checkpoint_steps'] == 0 or
                    global_step == config['max_steps'] or 
                    step == config['steps']
                ):
                    save_checkpoint(model, sampler, epoch, global_step,
                                    config['checkpoint_dir'])

                avg_loss = 0
                global_step += 1
                step += 1
                
                if (
                    global_step >= config['max_steps'] or 
                    step >= config['steps']
                ):
                    return global_step, perf_counter() - start_time
            else:
                model.step()

        epoch += 1


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
