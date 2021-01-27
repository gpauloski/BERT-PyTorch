import argparse
import os
import torch
import re


class Sample():
    def __init__(self, sentence, labels):
        assert len(sentence) == len(labels)
        self.sentence = sentence
        self.labels = labels

    def encoded(self, tokenizer, label_to_id):
        encoded_sentence = []
        encoded_labels = []
        for word, label in zip(self.sentence, self.labels):
            encoded = tokenizer.encode(word, add_special_tokens=False).ids
            label_ids = [label_to_id[label]] * len(encoded)
            encoded_sentence.extend(encoded)
            encoded_labels.extend(label_ids)
        assert len(encoded_sentence) == len(encoded_labels)
        return encoded_sentence, encoded_labels


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, filename, tokenizer, labels, max_seq_len):
        self.samples = self._parse_file(filename)
        self.tokenizer = tokenizer
        self.label_to_id = {label: i for i, label in enumerate(labels)}
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sentence, labels = self.samples[idx].encoded(
                self.tokenizer, self.label_to_id)
        mask = [1] * len(sentence)
        if len(sentence) > self.max_seq_len:
            sentence = sentence[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            mask = mask[:self.max_seq_len]
        else:
            while len(sentence) < self.max_seq_len:
                sentence.append(0)
                labels.append(0)
                mask.append(0)
        
        return torch.LongTensor(sentence), torch.LongTensor(mask), torch.LongTensor(labels)

    def _parse_file(self, filename):
        samples = []
        with open(filename, 'r') as f:
            sentence = []
            label = []
            for line in f:
                if line == '' or line.startswith('-DOCSTART') or line[0] == '\n':
                    if len(sentence) > 0:
                        samples.append(Sample(sentence, label))
                        sentence = []
                        label = []
                    continue
                tokens = re.split(' |\t', line)
                tokens = [t.strip() for t in tokens]
                sentence.append(tokens[0])
                label.append(tokens[3])
            if len(sentence) > 0:
                samples.append(Sample(sentence, label))

        return samples


if __name__ == '__main__':
    from bert.tokenization import get_wordpiece_tokenizer

    parser = argparse.ArgumentParser(description='Dataset test')
    parser.add_argument('--input_file', type=str, required=True,
                        help='CoNLL Input file')
    parser.add_argument('--vocab_file', type=str, required=True,
                        help='Vocab file for tokenizer')
    parser.add_argument('--max_seq_len', default=128, type=int,
                        help='Maximum sequence length')
    args = parser.parse_args()

    labels = ['O', 'I-DNA', 'B-DNA', 'I-RNA', 'B-RNA', 'I-cell_line', 'B-cell_line', 
              'I-protein', 'B-protein', 'I-cell_type', 'B-cell_type']
    tokenizer = get_wordpiece_tokenizer(args.vocab_file)
    dataset = NERDataset(args.input_file, tokenizer, labels, 
            max_seq_len=args.max_seq_len)

    for sample in dataset.samples:
        print(sample.sentence, sample.labels)
        print(sample.encoded(tokenizer, dataset.label_to_id))
        print('')

