import argparse
import h5py
import multiprocessing as mp
import os
import random
import time
import tokenizers

from pathlib import Path


class TrainingSample(object):
    def __init__(self, seq_tokens, next_seq_tokens=None, 
            is_random_next=False):
        self.seq_tokens = seq_tokens
        self.next_seq_tokens = next_seq_tokens
        self.is_random_next = is_random_next

        # next sentence: [CLS] sequence tokens [SEP] next sequence tokens [SEP] padding
        # no next sentence: [CLS] sequence tokens [SEP] padding
        self.sequence = ['[CLS]']
        self.special_token_positions = [0]
        self.sequence.extend(self.seq_tokens)
        if self.next_seq_tokens is not None:
            self.special_token_positions.append(len(self.sequence))
            self.sequence.append('[SEP]')
            self.sequence.extend(self.next_seq_tokens)
        self.special_token_positions.append(len(self.sequence))
        self.sequence.append('[SEP]')

    def __repr__(self):
        s = '(TrainingSample) {} (special_tokens={}, random_next={})'.format(
                self.sequence, self.special_token_positions,
                self.is_random_next)
        return s


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def get_documents_from_file(input_file, tokenizer):
    documents = [[]]
    with open(input_file, "r") as reader:
        for line in reader:
            line = convert_to_unicode(line).strip()
            if not line:
                # Empty lines are used as document delimiters so we start
                # a new document
                documents.append([])
            tokens = tokenizer.encode(line, add_special_tokens=False).tokens
            if tokens:
                documents[-1].append(tokens)
    # Remove empty documents
    documents = [x for x in documents if x]
    return documents


def create_samples_from_document(document_idx, documents, max_seq_len,
        next_seq_prob, short_seq_prob):
    samples = []
    chunk = []
    chunk_length = 0
    i = 0
    
    # Account for [CLS], [SEP], [SEP] tokens if doing next sentence pred
    # else just account for [CLS], [SEP]
    if next_seq_prob > 0:
        max_num_tokens = max_seq_len - 3
    else:
        max_num_tokens = max_seq_len - 2

    # Randomly reduce max seq length
    if random.random() < short_seq_prob:
        target_seq_length = random.randint(2, max_num_tokens)
    else:
        target_seq_length = max_num_tokens

    document = documents[document_idx]
    while i < len(document):
        current_seq = document[i]
        # It is possible the current sequence is greater than the allowed
        # max seq length so we have to clip those sequences
        if len(current_seq) > target_seq_length:
            current_seq = current_seq[:target_seq_length]
        # If out of sequences or adding next sequence would put us over target
        # save this chunk as a sample and reset the chunk
        if len(chunk) >= 1 and (i + 1 == len(document) or 
                chunk_length + len(current_seq) >= target_seq_length):
            if next_seq_prob > 0:
                if len(documents) <= 1:
                    raise ValueError('File only contained one document, '
                                     'Unable to make random next sequence.')
                # Divide chunk into two sequences
                seq_end = 1
                if len(chunk) >= 2:
                    seq_end = random.randint(1, len(chunk) - 1)

                seq_tokens = []
                for j in range(seq_end):
                    seq_tokens.extend(chunk[j])
                next_seq_tokens = []
                for j in range(seq_end, len(chunk)):
                    next_seq_tokens.extend(chunk[j])

                # If next sentence is random, overwrite next sequence
                # with a random one from a random article
                if random.random() < next_seq_prob:
                    is_random_next = True
                    next_seq_tokens = []
                    rand_idx = random.randint(0, len(documents) - 1)
                    while rand_idx == document_idx:
                        rand_idx = random.randint(0, len(documents) - 1)
                    rand_document = documents[rand_idx]
                    rand_start = random.randint(0, len(rand_document) - 1)
                    max_next_seq_len = target_seq_length - len(seq_tokens)
                    for j in range(rand_start, len(rand_document)):
                        next_seq_tokens.extend(rand_document[j])
                        if len(next_seq_tokens) >= max_next_seq_len:
                            next_seq_tokens = next_seq_tokens[:max_next_seq_len]
                    # We overwrote some of the last sequences with the random
                    # next seq so move the index backwards to reuse those
                    # overwritten sequences
                    i -= len(chunk) - seq_end
                else:
                    is_random_next = False
            else:
                seq_tokens = []
                for seq in chunk:
                    seq_tokens.extend(seq)
                next_seq_tokens = None
                is_random_next = False
          
            assert len(seq_tokens) <= target_seq_length
            if next_seq_tokens is not None:
                assert (len(next_seq_tokens) + len(seq_tokens) 
                        <= target_seq_length)

            samples.append(
                TrainingSample(seq_tokens, next_seq_tokens, is_random_next)
            )
            
            # Choose new random target len for next chunk
            if random.random() < short_seq_prob:
                target_seq_length = random.randint(2, max_num_tokens)
            else:
                target_seq_length = max_num_tokens
            
            chunk = []
            chunk_length = 0

        # We may have reset the chunk and updated the target_seq_length
        # or changed the index to get the current document again
        current_seq = document[i]
        if len(current_seq) > target_seq_length:
            current_seq = current_seq[:target_seq_length]
        chunk.append(current_seq)
        chunk_length += len(current_seq)
        i += 1

    return samples


def create_samples(input_file, tokenizer, max_seq_len, next_seq_prob,
        short_seq_prob):
    documents = get_documents_from_file(input_file, tokenizer)
    samples = []
    for i in range(len(documents)):
        samples.extend(
            create_samples_from_document(i, documents, max_seq_len, 
                    next_seq_prob, short_seq_prob)
        )
    random.shuffle(samples)
    return samples


def write_samples_to_hdf5(output_file, samples, tokenizer, max_seq_len):
    input_ids = []
    special_token_positions = []
    next_sent_labels = []
    while samples:
        # We pop here to save memory, we no longer need the sample after this
        sample = samples.pop()
        input_id = [tokenizer.token_to_id(tid) for tid in sample.sequence]
        #input_id = tokenizer.encode(sample.sequence, is_pretokenized=True, 
        #        add_special_tokens=False).ids
        assert len(input_id) == len(sample.sequence)
        assert None not in input_id
        while len(input_id) < max_seq_len:
            input_id.append(0)
        assert len(input_id) == max_seq_len, ('len(input_id)={}, max_seq_len={} '
                'len(sequence)={}'.format(
                len(input_id), max_seq_len, len(sample.sequence)))
        input_ids.append(input_id)
        special_token_positions.append(sample.special_token_positions)
        next_sent_labels.append(1 if sample.is_random_next else 0)
    
    with h5py.File(output_file, 'w') as f:
        f.create_dataset("input_ids", 
                data=input_ids, dtype='i4', compression='gzip')
        f.create_dataset("special_token_positions",
                data=special_token_positions, dtype='i4', compression='gzip')
        f.create_dataset("next_sentence_labels", 
                data=next_sent_labels, dtype='i1', compression='gzip')


def encode_file(input_file, output_file, tokenizer, max_seq_len,
            next_seq_prob, short_seq_prob):
    print("[encoder] Creating instances from {}".format(input_file))
    start_time = time.time()
    samples = create_samples(input_file, tokenizer, max_seq_len, next_seq_prob,
            short_seq_prob)
    write_samples_to_hdf5(output_file, samples, tokenizer, max_seq_len)
    print('[encoder] Encoded {} (time={:.0f}s)'.format(
            output_file, time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", default=None, type=str, required=True,
                        help='The input train corpus. Can be directory with '
                             '.txt files or a path to a single file')
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help='Output directory for hdf5 files')
    parser.add_argument("--vocab_file", default=None, type=str, required=True,
                        help="The vocabulary the BERT model will train on.")
    
    parser.add_argument("--max_seq_len", default=512, type=int,
                        help='The maximum total input sequence length after '
                             'WordPiece tokenization. Sequences longer than '
                             'this will be truncated, and sequences shorter '
                             'than this will be padded.')
    parser.add_argument("--short_seq_prob", default=0.1, type=float,
                        help='Probability to create a sequence shorter than '
                             'maximum sequence length')
    parser.add_argument("--next_seq_prob", default=0.0, type=float,
                        help='Probability to use a random next sentence.'
                             'If 0, skips next next sequence prediction')
    parser.add_argument("--uppercase", action='store_true', default=False,
                        help='Use uppercase.')
    parser.add_argument('--tokenizer', type=str, default='wordpiece',
                        choices=['wordpiece', 'bpe'],
                        help='Tokenizer to use')
    parser.add_argument('--processes', type=int, default=4,
                        help="Number of processes to use")

    args = parser.parse_args()
    
    overall_start_time = time.time()

    input_files = []
    if os.path.isfile(args.input_dir):
        input_files.append(args.input_dir)
    elif os.path.isdir(args.input_dir):
        for path in Path(args.input_dir).rglob('*.txt'):
            if path.is_file():
                input_files.append(str(path))
    else:
        raise ValueError("{} is not a valid path".format(args.input_file))

    print('[encoder] Found {} input files'.format(len(input_files)))

    output_prefix = 'sequences_'
    output_prefix += 'uppercase' if args.uppercase else 'lowercase'
    output_prefix += '_max_seq_len_' + str(args.max_seq_len)
    output_prefix += '_next_seq_task_' + str(
            True if args.next_seq_prob > 0 else False).lower()

    args.output_dir = os.path.join(args.output_dir, output_prefix)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    if args.tokenizer == 'wordpiece':
        tokenizer = tokenizers.BertWordPieceTokenizer(
            vocab=args.vocab_file,
            clean_text=True,
            handle_chinese_chars=True,
            lowercase=not args.uppercase,
        )
    elif args.tokenizer == 'bpe':
        tokenizer = tokenizers.ByteLevelBPETokenizer(
            vocab=args.vocab_file,
            add_prefix_space=True,
            lowercase=not args.uppercase,
            trim_offsets=True,
        )
    #tokenizer.enable_padding(length=args.max_seq_len)
    #tokenizer.enable_truncation(max_length=args.max_seq_len)

    params = []
    for i, ifile in enumerate(input_files):
        ofile = os.path.join(args.output_dir, 'train_{}.hdf5'.format(i))
        params.append((ifile, ofile, tokenizer, args.max_seq_len,
                args.next_seq_prob, args.short_seq_prob))

    print('[encoder] Starting multiprocessing pool ({} processes)'.format(
            args.processes))

    with mp.Pool(processes=args.processes) as pool:
        pool.starmap(encode_file, params)

    print('[encoder] Finished processing (time={:.0f}s)'.format(
            time.time() - overall_start_time))

