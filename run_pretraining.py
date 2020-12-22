# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT pretraining runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import h5py
import json
import loggerplus as logger
import math
import multiprocessing
import numpy as np
import os
import random
import signal
import time
import warnings
import torch
import kfac

from apex.optimizers import FusedLAMB
from torch.utils.data import DataLoader, RandomSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import bert.modeling as modeling
from bert.schedulers import PolyWarmUpScheduler
from bert.utils import is_main_process, get_world_size, get_rank, WorkerInitObj

from concurrent.futures import ProcessPoolExecutor

try:
    from torch.cuda.amp import autocast, GradScaler
    TORCH_FP16 = True
except:
    TORCH_FP16 = False

# Track whether a SIGTERM (cluster time up) has been handled
timeout_sent = False

# handle SIGTERM sent from the scheduler and mark so we
# can gracefully save & exit
def signal_handler(sig, frame):
    global timeout_sent
    timeout_sent = True

signal.signal(signal.SIGTERM, signal_handler)

def create_pretraining_dataset(input_file, max_pred_length, shared_list, args, worker_init):
    train_data = pretraining_dataset(input_file=input_file, max_pred_length=max_pred_length)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=args.local_batch_size, num_workers=4,
                                  worker_init_fn=worker_init, pin_memory=True)
    return train_dataloader, input_file


class pretraining_dataset(Dataset):
    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions',
                'masked_lm_ids', 'next_sentence_labels']
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        return len(self.inputs[0])

    def __getitem__(self, index):
        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5 else torch.from_numpy(
                np.asarray(input[index].astype(np.int64))) for indice, input in enumerate(self.inputs)]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = torch.nonzero(masked_lm_positions == 0, as_tuple=False)
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels]


class BertPretrainingCriterion(torch.nn.Module):
    def __init__(self, vocab_size):
        super(BertPretrainingCriterion, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size

    def forward(self, prediction_scores, seq_relationship_score, masked_lm_labels,
                next_sentence_labels):
        masked_lm_loss = self.loss_fn(prediction_scores.view(-1, self.vocab_size),
                                      masked_lm_labels.view(-1))
        next_sentence_loss = self.loss_fn(seq_relationship_score.view(-1, 2),
                                          next_sentence_labels.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss
        return total_loss


def parse_arguments():
    parser = argparse.ArgumentParser()

    ## Optional json config to override defaults below
    parser.add_argument("--config_file", default=None, type=str,
                        help="JSON config for overriding defaults")

    ## Required parameters. Note they can be provided in the json
    parser.add_argument("--input_dir", default=None, type=str,
                        help="The input data dir containing .hdf5 files for the task.")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output dir for checkpoints and logging.")
    parser.add_argument("--model_config_file", default=None, type=str,
                        help="The BERT model config")

    ## Training Configuration
    parser.add_argument('--disable_progress_bar', default=False, action='store_true',
                        help='Disable tqdm progress bar')
    parser.add_argument('--num_steps_per_checkpoint', type=int, default=100,
                        help="Number of update steps between writing checkpoints.")
    parser.add_argument('--skip_checkpoint', default=False, action='store_true',
                        help="Whether to save checkpoints")
    parser.add_argument('--checkpoint_activations', default=False, action='store_true',
                        help="Whether to use gradient checkpointing")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', default=False, action='store_true',
                        help="Use PyTorch AMP training")

    ## Hyperparameters
    parser.add_argument("--max_predictions_per_seq", default=80, type=int,
                        help="The maximum total of masked tokens in input sequence")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate.")
    parser.add_argument("--warmup_proportion", default=0.01, type=float,
                        help="Proportion of training to perform linear learning rate "
                             "warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument("--global_batch_size", default=2**16, type=int,
                        help="Global batch size for training.")
    parser.add_argument("--local_batch_size", default=8, type=int,
                        help="Per-GPU batch size for training.")
    parser.add_argument("--max_steps", default=1000, type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--previous_phase_end_step", default=0, type=int,
                        help="Final step of previous phase")

    ## KFAC Hyperparameters
    parser.add_argument('--kfac', default=False, action='store_true',
                        help='Use KFAC')
    parser.add_argument('--kfac_inv_interval', type=int, default=10,
                        help='iters between kfac inv ops')
    parser.add_argument('--kfac_factor_interval', type=int, default=1,
                        help='iters between kfac cov ops')
    parser.add_argument('--kfac_stat_decay', type=float, default=0.95,
                        help='Alpha value for covariance accumulation')
    parser.add_argument('--kfac_damping', type=float, default=0.003,
                        help='KFAC damping factor')
    parser.add_argument('--kfac_kl_clip', type=float, default=0.001,
                        help='KFAC KL gradient clip')
    parser.add_argument('--kfac_skip_layers', nargs='+', type=str, 
                        default=['BertLMPredictionHead', 'embedding'],
                        help='Modules to ignore registering with KFAC '
                             '(default: [BertLMPredictionHead, embedding])')

    # Set by torch.distributed.launch
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training')

    args = parser.parse_args()

    if args.config_file is not None:
        with open(args.config_file) as jf:
            configs = json.load(jf)
        for key in configs:
            if key in vars(args):
                setattr(args, key, configs[key])

    return args


def setup_training(args):
    assert (torch.cuda.is_available())

    torch.cuda.set_device(args.local_rank)
    args.device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)

    logger.init(
        handlers=[
            logger.StreamHandler(verbose=is_main_process()),
            logger.FileHandler(os.path.join(args.output_dir, 'log.txt'),
                               overwrite=False, verbose=is_main_process()),
            logger.TorchTensorboardHandler(args.output_dir,
                                           verbose=is_main_process()),
            #logger.CSVHandler(os.path.join(args.output_dir, 'metrics.csv'),
            #                   overwrite=False),
        ]
    )

    logger.info('Torch distributed initialized (world_size={}, backend={})'.format(
            get_world_size(), torch.distributed.get_backend()))

    if not TORCH_FP16 and args.fp16:
        raise ValueError('FP16 training enabled but unable to import torch.cuda.amp.'
                         'Is the torch version >= 1.6?')

    if args.global_batch_size % get_world_size() != 0 and is_main_process():
        warnings.warn('global_batch_size={} is not divisible by world_size={}.'
                      ' The last batch will be padded with additional '
                      'samples.'.format(
                      args.global_batch_size, get_world_size()))
    args.local_accumulated_batch_size = math.ceil(
            args.global_batch_size / get_world_size())

    if args.local_accumulated_batch_size % get_world_size() != 0 and is_main_process():
        raise ValueError('local_accumulated_batch_size={} is not divisible '
                         'by local_batch_size={}. local_accumulated_batch_size '
                         'is global_batch_size // world_size. The last '
                         'batch will be padded with additional samples'.format(
                         args.local_accumulated_batch_size, get_world_size()))
    args.accumulation_steps = math.ceil(
            args.local_accumulated_batch_size / args.local_batch_size)

    return args


def prepare_model(args):
    config = modeling.BertConfig.from_json_file(args.model_config_file)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
    model = modeling.BertForPreTraining(config)

    checkpoint = None
    global_step = 0
    args.resume_step = 0
    checkpoint_names = [f for f in os.listdir(args.output_dir) if f.endswith(".pt")]
    if len(checkpoint_names) > 0:
        args.resume_step = max([int(x.split('.pt')[0].split('_')[1].strip())
                           for x in checkpoint_names])

        checkpoint = torch.load(
                os.path.join(args.output_dir, "ckpt_{}.pt".format(args.resume_step)),
                map_location="cpu"
        )

        model.load_state_dict(checkpoint['model'], strict=False)

        if args.previous_phase_end_step > args.resume_step:
            raise ValueError('previous_phase_end_step={} cannot be larger '
                             'than resume_step={}'.format(
                             args.previous_phase_end_step, args.resume_step))
        global_step = args.resume_step - args.previous_phase_end_step
        
        logger.info('Resume from step {} checkpoint'.format(args.resume_step))

    model.to(args.device)
    model.checkpoint_activations(args.checkpoint_activations)

    if get_world_size() > 1:
        model = DDP(model, device_ids=[args.local_rank])

    criterion = BertPretrainingCriterion(config.vocab_size)

    return model, checkpoint, global_step, criterion, args


def prepare_optimizers(args, model, checkpoint, global_step):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = FusedLAMB(optimizer_grouped_parameters,
                          lr=args.learning_rate)
    lr_schedulers = [PolyWarmUpScheduler(optimizer,
                                         warmup=args.warmup_proportion,
                                         total_steps=args.max_steps)]

    scaler = None
    if args.fp16:
        scaler = GradScaler()

    preconditioner = None
    if args.kfac:
        preconditioner = kfac.KFAC(
            model,
            lr=args.learning_rate, 
            factor_decay=args.kfac_stat_decay,
            damping=args.kfac_damping, 
            kl_clip=args.kfac_kl_clip,
            batch_first=True,
            factor_update_freq=args.kfac_factor_interval,
            inv_update_freq=args.kfac_inv_interval,
            use_eigen_decomp=True,
            # Skip TrainingHeads which contains the decoder, a Linear module
            # with shape (seq_len, vocab_size), such that it is too large to invert
            skip_layers=args.kfac_skip_layers,
            # False b/c we use grad accumulation, accumulating the input/output
            # data will substantially increase memory usage
            accumulate_data=False,
            # BERT calls KFAC very infrequently so no need to optimize for
            # communication. Optimize for memory instead.
            comm_method=kfac.CommMethod.MEM_OPT,
            # Compute the factors and update the running averages during the
            # forward backward pass b/c we are using grad accumulation but
            # not accumulating the input/output data
            compute_factor_in_hook=True,
            distribute_layer_factors=False,
            grad_scaler=scaler,
            verbose=False,
        )
        lrs = PolyWarmUpScheduler(
            preconditioner, 
            warmup=args.warmup_proportion, 
            total_steps=args.max_steps
        )
        lr_schedulers.append(lrs)

        if is_main_process():
            logger.info(preconditioner)

    if checkpoint is not None:
        if args.resume_step >= args.previous_phase_end_step:
            keys = list(checkpoint['optimizer']['state'].keys())
            # Override hyperparameters from previous checkpoint
            for key in keys:
                checkpoint['optimizer']['state'][key]['step'] = global_step
            for iter, item in enumerate(checkpoint['optimizer']['param_groups']):
                checkpoint['optimizer']['param_groups'][iter]['step'] = global_step
                checkpoint['optimizer']['param_groups'][iter]['t_total'] = args.max_steps
                checkpoint['optimizer']['param_groups'][iter]['warmup'] = args.warmup_proportion
                checkpoint['optimizer']['param_groups'][iter]['lr'] = args.learning_rate
        optimizer.load_state_dict(checkpoint['optimizer'])
        if ('preconditioner' in checkpoint and
                checkpoint['preconditioner'] is not None and
                preconditioner is not None):
            preconditioner.load_state_dict(checkpoint['preconditioner'])

    return optimizer, preconditioner, lr_schedulers, scaler


def take_optimizer_step(optimizer, preconditioner, model, scaler):
    if preconditioner is not None:
        if scaler is not None:
            scaler.unscale_(optimizer)
        preconditioner.step()
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    for param in model.parameters():
        param.grad = None


def forward_backward_pass(model, criterion, scaler, batch, divisor,
                          sync_grads=True):
    input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch

    if scaler is not None:
        with autocast():
            prediction_scores, seq_relationship_score = model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask)
            loss = criterion(
                    prediction_scores,
                    seq_relationship_score,
                    masked_lm_labels,
                    next_sentence_labels)
    else:
        prediction_scores, seq_relationship_score = model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask)
        loss = criterion(
                prediction_scores,
                seq_relationship_score,
                masked_lm_labels,
                next_sentence_labels)

    loss = loss / divisor

    if not sync_grads:
        with model.no_sync():
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
    else:
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    return loss


def main(args):
    global timeout_sent

    model, checkpoint, global_step, criterion, args = prepare_model(args)
    optimizer, preconditioner, lr_schedulers, scaler = prepare_optimizers(
            args, model, checkpoint, global_step)

    model.train()
    most_recent_ckpts_paths = []
    average_loss = 0.0
    epoch = 0
    training_steps = 0
    train_time_start = None

    worker_init = WorkerInitObj(args.seed + args.local_rank)
    pool = ProcessPoolExecutor(1)

    # Note: We loop infinitely over epochs, termination is handled via iteration count
    while True:
        restored_data_loader = None
        if checkpoint is None or epoch > 0 or global_step == 0:
            files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if
                     os.path.isfile(os.path.join(args.input_dir, f)) and 'training' in f]
            files.sort()
            num_files = len(files)
            random.Random(args.seed + epoch).shuffle(files)
            f_start_id = 0
        else:
            f_start_id = checkpoint['files'][0]
            files = checkpoint['files'][1:]
            args.resume_from_checkpoint = False
            num_files = len(files)
            epoch = checkpoint.get('epoch', 0)
            restored_data_loader = checkpoint.get('data_loader', None)

        shared_file_list = {}

        if get_world_size() > num_files:
            remainder = get_world_size() % num_files
            data_file = files[(f_start_id*get_world_size()+get_rank() + remainder*f_start_id)%num_files]
        else:
            data_file = files[(f_start_id*get_world_size()+get_rank())%num_files]

        previous_file = data_file

        if restored_data_loader is None:
            train_data = pretraining_dataset(data_file, args.max_predictions_per_seq)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                          batch_size=args.local_batch_size,
                                          num_workers=4, worker_init_fn=worker_init,
                                          pin_memory=True)
        else:
            train_dataloader = restored_data_loader
            restored_data_loader = None


        for f_id in range(f_start_id + 1 , len(files)):

            if get_world_size() > num_files:
                data_file = files[(f_id*get_world_size()+get_rank() + remainder*f_id)%num_files]
            else:
                data_file = files[(f_id*get_world_size()+get_rank())%num_files]

            previous_file = data_file

            dataset_future = pool.submit(create_pretraining_dataset, data_file,
                    args.max_predictions_per_seq, shared_file_list, args, worker_init)

            train_iter = tqdm(train_dataloader, desc="Iteration",
                    disable=not (not args.disable_progress_bar and is_main_process()))

            if train_time_start is None:
                train_time_start = time.time()

            for batch in train_iter:
                training_steps += 1

                batch = [t.to(args.device) for t in batch]
                loss = forward_backward_pass(model, criterion, scaler, batch,
                        args.accumulation_steps,
                        sync_grads=training_steps % args.accumulation_steps==0)
                average_loss += loss.item()

                if training_steps % args.accumulation_steps == 0:
                    for lrs in lr_schedulers:
                        lrs.step()
                    take_optimizer_step(optimizer, preconditioner, 
                            model, scaler)
                    global_step += 1

                if training_steps % args.accumulation_steps == 0:
                    logger.log(tag='train',
                               step=global_step+args.previous_phase_end_step,
                               epoch=epoch,
                               average_loss=average_loss,
                               step_loss=loss.item() * args.accumulation_steps,
                               learning_rate=optimizer.param_groups[0]['lr'])
                    average_loss = 0

                if (global_step >= args.max_steps or
                        training_steps % (args.num_steps_per_checkpoint * args.accumulation_steps) == 0 or
                        timeout_sent):
                    if global_step >= args.max_steps or timeout_sent:
                        final_time = time.time() - train_time_start
                    if is_main_process() and not args.skip_checkpoint:
                        # Save a trained model
                        logger.info('Saving checkpoint: global_step={}'.format(
                                global_step + args.previous_phase_end_step))
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_save_file = os.path.join(args.output_dir,
                                "ckpt_{}.pt".format(global_step + args.previous_phase_end_step))
                        torch.save(
                            {
                                'model': model_to_save.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'preconditioner': preconditioner.state_dict()
                                        if preconditioner is not None else None,
                                'files': [f_id] + files,
                                'epoch': epoch,
                                'data_loader': None if global_step >= args.max_steps else train_dataloader
                            },
                            output_save_file
                        )

                        most_recent_ckpts_paths.append(output_save_file)
                        if len(most_recent_ckpts_paths) > 3:
                            ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
                            os.remove(ckpt_to_be_removed)

                # Exiting the training due to hitting max steps, or being sent a
                # timeout from the cluster scheduler
                if global_step >= args.max_steps or timeout_sent:
                    del train_dataloader
                    return global_step, final_time

            del train_dataloader
            train_dataloader, data_file = dataset_future.result(timeout=None)

        epoch += 1


if __name__ == "__main__":
    args = parse_arguments()
    
    if args.input_dir is None:
        raise ValueError('--input_dir must be provided via arguments or the '
                         'config file')
    if args.output_dir is None:
        raise ValueError('--output_dir must be provided via arguments or the '
                         'config file')
    if args.model_config_file is None:
        raise ValueError('--model_config_file must be provided via arguments '
                         'or the config file')

    random.seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    torch.manual_seed(args.seed + args.local_rank)
    torch.cuda.manual_seed(args.seed + args.local_rank)

    args = setup_training(args)
    logger.info(args)

    start_time = time.time()
    global_step, train_time = main(args)
    runtime = time.time() - start_time

    logger.info("runtime: {}  train_time: {}  training_seq_per_sec: {}".format(
            runtime, train_time,
            args.global_batch_size * global_step / train_time))

