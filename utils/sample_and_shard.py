import argparse
import os
import codecs
import time
import random

from pathlib import Path


def parse_value_as_int(value):
    postfix_map = {'K': 1000, 'M': 1000000, 'B': 1000000000}
    if isinstance(value, int) or isinstance(value, float):
        return int(value)
    if value.isdigit():
        return int(value)
    if len(value) > 1:
        return int(float(value[:-1]) * postfix_map[value[-1].upper()])
    raise ValueError('Unable to parse "{}" as integer'.format(value))


def file_to_articles(filepath):
    articles = [[]]

    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if line == '':
                articles.append([])
            else:
                articles[-1].append(line)

    # Remove blank documents
    articles = [a for a in articles if a]

    return articles


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text file sharder')

    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Directory of input .txt files')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output directory to write shards to')
    parser.add_argument('-f', '--format', type=str, default='shard_{index}.txt',
                        help='Output file name format. {index} will be '
                             'with the numeric index of the output shard')
    parser.add_argument('-b', '--size', type=str, required=True,
                        help='Maximum number of bytes per shard')
    parser.add_argument('-n', '--sentences', type=str, required=True,
                        help='Number of sentences in total to sample')
    args = parser.parse_args()

    print('[sampler] Sharding {} to {}'.format(args.input, args.output))

    overall_start_time = time.time()

    input_files = []
    if os.path.isfile(args.input):
        input_files.append(args.input)
    elif os.path.isdir(args.input):
        for path in Path(args.input).rglob('*.txt'):
            if path.is_file():
                input_files.append(str(path))
    else:
        raise ValueError("{} is not a valid path".format(args.input_file))

    print('[sampler] Found {} input files'.format(len(input_files)))

    sentences = parse_value_as_int(args.sentences)
    shard_size = parse_value_as_int(args.size)

    # number of sentences to sample from each input_file
    sentences_per_input = sentences // len(input_files)

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    shard_idx = 0
    ofile_format = os.path.join(args.output, args.format)
    ofile = open(ofile_format.format(index=shard_idx), 'w', encoding='utf-8')

    for i, filepath in enumerate(input_files):
        start = time.time()
        articles = file_to_articles(filepath)
        print('Extracting articles: {}s'.format(time.time() - start))

        # Randomly select articles until we reach the per file sentence count target
        start_ = time.time()
        indices = list(range(len(articles)))
        random.shuffle(indices)
        selected = []
        sentence_count = 0
        while sentence_count < sentences_per_input and len(indices) > 0:
            idx = indices.pop()
            selected.append(idx)
            sentence_count += len(articles[idx])

        articles = [articles[i] for i in selected]
        print('Sampling articles: {}s'.format(time.time() - start_))

        start_ = time.time()
        for article in articles:
            # If current open file has reached file size limit, close it
            # and open a new one
            if ofile.tell() > shard_size:
                ofile.close()
                shard_idx += 1
                ofile = open(ofile_format.format(index=shard_idx), 'w', encoding='utf-8')
            # Write entire article to current file, one sentence per line
            for line in article:
                ofile.write(line + '\n')
            # Add blank line to end of article
            ofile.write('\n')
        print('Writing articles: {}s'.format(time.time() - start_))

        print('[sampler] Finished sampling from input file {}/{} (time={})'.format(
                i+1, len(input_files), time.time() - start))

        del articles

    ofile.close()

    print('[sampler] Finished sampling and sharding (time={})'.format(
            time.time() - overall_start_time))
