import argparse
import multiprocessing as mp
import os
import shutil
import subprocess
import time
import warnings

from pathlib import Path
from nltk.tokenize import sent_tokenize


def get_sentences(lines):
    text = ' '.join(lines)
    text = text.replace('\n', ' ')
    return [s.strip() for s in sent_tokenize(text)]


class Formatter():
    def __init__(self, name, input_dir, output_dir):
        self.name = name
        self.input_dir = input_dir
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def format(self, processes=1, shards=-1):
        # if shards < 1, default shards to number of input files
        print('[{}] Searching for input files in {}'.format(
                self.name, self.input_dir))

        files = []
        for path in Path(self.input_dir).rglob('*'):
            if path.is_file():
                files.append(str(path))
        
        if len(files) == 0:
            raise RuntimeError('Found 0 files in {}'.format(input_dir))

        if shards < 1:
            shards = len(files)

        if shards > len(files):
            shards = len(files)

        print('[{}] Dividing {} input files across {} shards'.format(
                self.name, len(files), shards))

        params = []
        for i in range(shards):
            output_file = os.path.join(self.output_dir, 
                    '{}_one_sentence_per_line_{}.txt'.format(self.name, i))
            params.append(([], output_file))

        # round robin assign input files to shards
        for i, f in enumerate(files):
            params[i % len(params)][0].append(f)

        print('[{}] Starting multiprocessing pool ({} processes)'.format(
                self.name, processes))
        with mp.Pool(processes=processes) as pool:
            pool.starmap(self._format, params)


class WikiCorpusFormatter(Formatter):
    def __init__(self, input_dir, output_dir):
        super(WikiCorpusFormatter, self).__init__(
                'wikicorpus', input_dir, output_dir)

    def preprocess(self, processes=1, shard_size='100M'):
        # Note: this function is not called anywhere and has been
        # replaces with just calling wikiextractor from the terminal
        # but I am leaving this here because it works

        # Use wikiextractor to parse .xml file from Wikipedia
        # Writes files to self.input_dir/data
        xml_file = os.path.join(self.input_dir, 'wikicorpus_en.xml')
        if not os.path.exists(xml_file):
            raise ValueError('Unable to find wikicorpus_en.xml in '
                             '{}'.format(xml_file))
        output_dir = os.path.join(self.input_dir, 'data')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            warnings.warn('WikiExtractor output directory ({}) already '
                          'exists. Skipping WikiExtractor. If this is'
                          'a mistake, delete data/ and run again'.format(
                          output_dir))
            return
        cmd = 'wikiextractor {} -b {} '.format(xml_file, shard_size)
        cmd += '--processes {} -o {} '.format(processes, output_dir)
        print('[{}] Running WikiExtractor'.format(self.name))
        subprocess.run(cmd, shell=True, check=True)
        print('[{}] WikiExtractor finished'.format(self.name))

    def _format(self, input_files, output_file):
        start_time = time.time()
        with open(output_file, mode='w', encoding='utf-8') as ofile:
            for input_file in input_files:
                with open(input_file, mode='r') as ifile:
                    article_open = False
                    article_lines = []
                    for line in ifile:
                        if line.startswith('<doc id='):
                            article_open = True
                        elif line.startswith('</doc>'):
                            # First line in article_lines is always the title
                            sentences = get_sentences(article_lines[1:])
                            for s in sentences:
                                ofile.write(s.strip() + '\n')
                            ofile.write('\n')
                            article_open = False
                            article_lines = []
                        elif article_open:
                            article_lines.append(line)
        print('[{}] Finished shard: {} (time={:.1f}s)'.format(
                self.name, output_file, time.time() - start_time))


class BooksCorpusFormatter(Formatter):
    def __init__(self, input_dir, output_dir):
        super(BooksCorpusFormatter, self).__init__(
                'bookscorpus', input_dir, output_dir)

    def _format(self, input_files, output_file):
        start_time = time.time()
        with open(output_file, mode='w', encoding='utf-8') as ofile:
            for input_file in input_files:
                try:
                    with open(input_file, mode='r') as ifile:
                        sentences = get_sentences(ifile.read())
                        for s in sentences:
                            ofile.write(s.strip() + '\n')
                        ofile.write('\n')
                except:
                    print('[{}] Unable to format {}'.format(
                            self.name, input_file))
        print('[{}] Finished shard: {} (time={:.1f}s)'.format(
                self.name, output_file, time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Format and shard datasets into one sentence per '
                        'line and articles separated by blank lines.')

    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing dataset downloaded with '
                             'data/download.py')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to write sharded/formatted data to')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['wikicorpus', 'bookscorpus'],
                        help='Dataset to format')
    parser.add_argument('--processes', type=int, default=8,
                        help='Number of processes to use for parallel tasks')
    parser.add_argument('--shards', type=int, default=64,
                        help='Number of shards for dataset')
    args = parser.parse_args()

    print('Formatting {} ({} processes)'.format(args.dataset, args.processes))
    start_time = time.time()

    if args.dataset == 'wikicorpus':
        formatter = WikiCorpusFormatter(args.input_dir, args.output_dir)
    elif args.dataset == 'bookscorpus':
        formatter = BooksCorpusFormatter(args.input_dir, args.output_dir)
    else:
        raise ValueError('Unknown dataset "{}"'.format(args.dataset))

    formatter.format(processes=args.processes, shards=args.shards)

    print('Finished formatting (time={:.0f}s)'.format(
            time.time() - start_time))

