import argparse
import json
import os
import torch

import src.modeling as modeling
import src.tokenization as tokenization
import numpy as np
import torch.nn.functional as F

from apex.optimizers import FusedAdam
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
from tqdm import tqdm
from src.ner_dataset import NERDataset


def parse_arguments():
    parser = argparse.ArgumentParser()

    ## Required parameters. Note they can be provided in the json
    parser.add_argument("--train_file", type=str, required=True,
                        help="Training data file in CoNLL format")
    parser.add_argument("--val_file", default=None, type=str,
                        help="Validation data file")
    parser.add_argument("--test_file", default=None, type=str,
                        help="Test data file")

    parser.add_argument("--labels", type=str, nargs='+',
                        help="Entity labels")
    parser.add_argument("--model_config_file", type=str, required=True,
                        help="The BERT model config")
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Path to pretrained model checkpoint")
    parser.add_argument("--vocab_file", default=None, type=str, required=False,
                        help="The vocab file used for training the model"
                             "Can be passed via model_config_file instead")
    parser.add_argument("--uppercase", default=False, action="store_true",
                        help="Use uppercase model and tokenizer")
    parser.add_argument('--tokenizer', type=str, default=None,
                        choices=['wordpiece', 'bpe'],
                        help='Tokenization Method')

    parser.add_argument('--epochs', type=int, default=10,
                        help="random seed for initialization")
    parser.add_argument('--lr', type=float, default=0.2,
                        help="random seed for initialization")
    parser.add_argument('--clip_grad', type=float, default=5.0,
                        help="Gradient norm clipping")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size")
    parser.add_argument('--max_seq_len', type=int, default=512,
                        help="Max sequence length")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    return args


def get_data(args):
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    kwargs['batch_size'] = args.batch_size

    if args.vocab_file is None:
        with open(args.model_config_file) as f:
            configs = json.load(f)
            if 'vocab_file' not in configs:
                raise ValueError('vocab_file must be provided in the model '
                                 'config or as a command line option')
            args.vocab_file = configs['vocab_file'] 
    
    if args.tokenizer is None:
        with open(args.model_config_file) as f:
            configs = json.load(f)
            if 'tokenizer' not in configs:
                raise ValueError('tokenizer must be provided in the model '
                                 'config or as a command line option')
            args.tokenizer = configs['tokenizer'] 

    if args.tokenizer == 'wordpiece':
        tokenizer = tokenization.get_wordpiece_tokenizer(
                args.vocab_file, uppercase=args.uppercase)
    elif args.tokenizer == 'bpe':
        tokenizer = tokenization.get_bpe_tokenizer(
                args.vocab_file, uppercase=args.uppercase)

    train_dataset = NERDataset(args.train_file, tokenizer, args.labels,
            max_seq_len=args.max_seq_len)
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, 
            sampler=train_sampler, **kwargs)

    if args.val_file is not None:
        val_dataset = NERDataset(args.val_file, tokenizer, args.labels,
                max_seq_len=args.max_seq_len)
        val_loader = torch.utils.data.DataLoader(val_dataset, **kwargs)
    else:
        val_loader = None

    if args.test_file is not None:
        test_dataset = NERDataset(args.test_file, tokenizer, args.labels,
                max_seq_len=args.max_seq_len)
        test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)
    else:
        test_loader = None

    return train_loader, val_loader, test_loader


class Metric():
    def __init__(self):
        self.total = 0
        self.n = 0

    def update(self, value):
        self.total += value
        self.n += 1

    @property
    def avg(self):
        return self.total / self.n


def train(model, optimizer, train_loader, epoch, args):
    model.train()
    loss_metric = Metric()
    loss_fn = torch.nn.CrossEntropyLoss()

    with tqdm(total=len(train_loader),
              bar_format='{l_bar}{bar:8}{r_bar}',
              desc='Epoch {}/{}'.format(epoch, args.epochs)) as t:
        for batch in train_loader:
            if args.cuda:
                batch = [b.cuda() for b in batch]
            
            optimizer.zero_grad()
            sequences, labels, masks = batch
            loss = model(sequences, token_type_ids=torch.zeros_like(masks), #None, 
                    attention_mask=masks, labels=labels)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters=model.parameters(), 
                    max_norm=args.clip_grad)
            optimizer.step()
            
            loss_metric.update(loss.item())

            t.set_postfix_str('train_loss: {:.5f}, lr: {:.2e}'.format(
                    loss_metric.avg, optimizer.param_groups[0]['lr']))
            t.update(1)


@torch.no_grad()
def evaluate(model, data_loader, args):
    model.eval()
    
    loss_metric = Metric()
    loss_fn = torch.nn.CrossEntropyLoss()

    predictions = []
    true_labels = []

    for batch in data_loader:
        if args.cuda:
            batch = [b.cuda() for b in batch]
        sequences, labels, masks = batch
        loss = model(sequences, token_type_ids=torch.zeros_like(masks), 
                attention_mask=masks, labels=labels)
        # pred is shape (batch_size, max_seq_len, n_labels)
        logits = model(sequences, token_type_ids=torch.zeros_like(masks), 
                attention_mask=masks)

        loss_metric.update(loss)
        
        logits = logits.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend([list(l) for l in labels])

    tag_values = {i: tag for i, tag in enumerate(args.labels)}

    pred_tags = []
    valid_tags = []

    for p, l in zip(predictions, true_labels):
        for p_i, l_i in zip(p, l):
            if l_i > 0:
                pred_tags.append(tag_values[p_i])
                valid_tags.append(tag_values[l_i])

    return loss_metric.avg, f1_score(pred_tags, valid_tags, average='macro')


if __name__ == '__main__':
    args = parse_arguments()

    torch.manual_seed(args.seed)
    args.cuda = torch.cuda.is_available()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    ## MODEL
    config = modeling.BertConfig.from_json_file(args.model_config_file)
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
    model = modeling.BertForTokenClassification(config, len(args.labels))
    model.load_state_dict(
            torch.load(args.model_checkpoint, map_location='cpu')['model'], 
            strict=False)
    model.to('cuda' if args.cuda else 'cpu')

    ## OPTIMIZER
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    params = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in params if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = FusedAdam(optimizer_grouped_parameters, lr=args.lr, 
                          bias_correction=False)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(1 + 0.05*epoch))

    ## DATA
    train_loader, val_loader, test_loader = get_data(args)

    for epoch in range(args.epochs):
        train(model, optimizer, train_loader, epoch, args)
        if val_loader is not None:
            loss, f1 = evaluate(model, val_loader, args)
            print('val_loss: {:.5f}, val_f1: {:.5f}'.format(loss, f1))
        scheduler.step()

    if test_loader is not None:
        loss, f1 = evaluate(model, test_loader, args)
        print('test_loss: {:.5f}, test_f1: {:.5f}'.format(loss, f1))

