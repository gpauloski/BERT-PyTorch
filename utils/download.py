import argparse
import bz2
import hashlib
import os
import urllib.request
import subprocess
import sys
import zipfile


class Downloader():
    def __init__(self, save_path):
        self.save_path = save_path

    def download(self, name):
        if name == 'bookscorpus':
            downloader = BooksCorpusDownloader(self.save_path)
        elif name == 'mprc':
            downloader = GLUEDownloader(self.save_path, tasks=['MPRC'])
        elif name == 'sst-2':
            downloader = GLUEDownloader(self.save_path, tasks=['SST'])
        elif name == 'squad':
            downloader = SquadDownloader(self.save_path)
        elif name == 'wikicorpus':
            downloader = WikiCorpusDownloader(self.save_path)
        elif name == 'weights':
            downloader = WeightsDownloader(self.save_path)
        else:
            raise ValueError('Unknown datasets "{}"'.format(name))

        downloader.download()
        downloader.extract()


class DatasetDownloader():
    def __init__(self, save_path, name):
        self.save_path = os.path.join(save_path, name)
        self.name = name

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def download(self):
        for url, dst in self.download_urls.items():
            dst_path = os.path.join(self.save_path, dst)
            print('[{}] Downloading: {}'.format(self.name, url))
            if os.path.isfile(dst_path):
                print('[{}] ** Download file already exists, skipping '
                      'download'.format(self.name))
            else:
                response = urllib.request.urlopen(url)
                with open(dst_path, "wb") as handle:
                    handle.write(response.read())

    def extract(self):
        pass


class BooksCorpusDownloader(DatasetDownloader):
    def __init__(self, save_path):
        super(BooksCorpusDownloader, self).__init__(save_path, 'bookscorpus')

    def download(self):
        bookcorpus_repo = os.path.join(self.save_path, 'bookcorpus')
        if os.path.exists(bookcorpus_repo):
            print('[{}] Bookcorpus repository already exists, skipping'.format(
                    self.name))
        else:
            subprocess.run(
                'git clone https://github.com/soskek/bookcorpus.git {}'.format(
                bookcorpus_repo), shell=True, check=True)

        download_path = os.path.join(self.save_path, 'data')
        print('[{}] Calling download_files.py'.format(self.name))
        cmd = 'python {}/download_files.py --list {}/url_list.jsonl '.format(
                bookcorpus_repo, bookcorpus_repo)
        cmd += '--out {} --trash-bad-count'.format(download_path)
        subprocess.run(cmd, shell=True, check=True)
        

class GLUEDownloader(DatasetDownloader):
    def __init__(self, save_path, tasks=['MRPC', 'SST']):
        super(GLUEDownloader, self).__init__(save_path, 'glue')

        # Task options: ["CoLA", "SST", "MRPC", "QQP", "STS", "MNLI", "SNLI", 
        #                "QNLI", "RTE", "WNLI", "diagnostic"]
        self.tasks = tasks

        self.download_urls = {
            'https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py' : 'download_glue_data.py'
        }

    def download(self):
        super(GLUEDownloader, self).download()
        sys.path.append(self.save_path)
        import download_glue_data
        for task in self.tasks:
            download_glue_data.main(
                    ['--data_dir', self.save_path, '--tasks', task])
        sys.path.pop()


class SquadDownloader(DatasetDownloader):
    def __init__(self, save_path):
        super(SquadDownloader, self).__init__(save_path, 'squad')

        if not os.path.exists(self.save_path + '/v1.1'):
            os.makedirs(self.save_path + '/v1.1')

        if not os.path.exists(self.save_path + '/v2.0'):
            os.makedirs(self.save_path + '/v2.0')

        self.download_urls = {
            'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json' : 'v1.1/train-v1.1.json',
            'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json' : 'v1.1/dev-v1.1.json',
            'https://worksheets.codalab.org/rest/bundles/0xbcd57bee090b421c982906709c8c27e1/contents/blob/' : 'v1.1/evaluate-v1.1.py',
            'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json' : 'v2.0/train-v2.0.json',
            'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json' : 'v2.0/dev-v2.0.json',
            'https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/' : 'v2.0/evaluate-v2.0.py',
        }


class WeightsDownloader(DatasetDownloader):
    def __init__(self, save_path):
        super(WeightsDownloader, self).__init__(
                save_path, 'google_pretrained_weights')

        # Download urls
        self.model_urls = {
            'bert_base_uncased': ('https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip', 'uncased_L-12_H-768_A-12.zip'),
            'bert_large_uncased': ('https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip', 'uncased_L-24_H-1024_A-16.zip'),
            'bert_base_cased': ('https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip', 'cased_L-12_H-768_A-12.zip'),
            'bert_large_cased': ('https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip', 'cased_L-24_H-1024_A-16.zip'),
        }
      
        # SHA256sum verification for file download integrity (and checking for changes from the download source over time)
        self.bert_base_uncased_sha = {
            'bert_config.json': '7b4e5f53efbd058c67cda0aacfafb340113ea1b5797d9ce6ee411704ba21fcbc',
            'bert_model.ckpt.data-00000-of-00001': '58580dc5e0bf0ae0d2efd51d0e8272b2f808857f0a43a88aaf7549da6d7a8a84',
            'bert_model.ckpt.index': '04c1323086e2f1c5b7c0759d8d3e484afbb0ab45f51793daab9f647113a0117b',
            'bert_model.ckpt.meta': 'dd5682170a10c3ea0280c2e9b9a45fee894eb62da649bbdea37b38b0ded5f60e',
            'vocab.txt': '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3',
        }

        self.bert_large_uncased_sha = {
            'bert_config.json': 'bfa42236d269e2aeb3a6d30412a33d15dbe8ea597e2b01dc9518c63cc6efafcb',
            'bert_model.ckpt.data-00000-of-00001': 'bc6b3363e3be458c99ecf64b7f472d2b7c67534fd8f564c0556a678f90f4eea1',
            'bert_model.ckpt.index': '68b52f2205ffc64dc627d1120cf399c1ef1cbc35ea5021d1afc889ffe2ce2093',
            'bert_model.ckpt.meta': '6fcce8ff7628f229a885a593625e3d5ff9687542d5ef128d9beb1b0c05edc4a1',
            'vocab.txt': '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3',
        }

        self.bert_base_cased_sha = {
            'bert_config.json': 'f11dfb757bea16339a33e1bf327b0aade6e57fd9c29dc6b84f7ddb20682f48bc',
            'bert_model.ckpt.data-00000-of-00001': '734d5a1b68bf98d4e9cb6b6692725d00842a1937af73902e51776905d8f760ea',
            'bert_model.ckpt.index': '517d6ef5c41fc2ca1f595276d6fccf5521810d57f5a74e32616151557790f7b1',
            'bert_model.ckpt.meta': '5f8a9771ff25dadd61582abb4e3a748215a10a6b55947cbb66d0f0ba1694be98',
            'vocab.txt': 'eeaa9875b23b04b4c54ef759d03db9d1ba1554838f8fb26c5d96fa551df93d02',
        }

        self.bert_large_cased_sha = {
            'bert_config.json': '7adb2125c8225da495656c982fd1c5f64ba8f20ad020838571a3f8a954c2df57',
            'bert_model.ckpt.data-00000-of-00001': '6ff33640f40d472f7a16af0c17b1179ca9dcc0373155fb05335b6a4dd1657ef0',
            'bert_model.ckpt.index': 'ef42a53f577fbe07381f4161b13c7cab4f4fc3b167cec6a9ae382c53d18049cf',
            'bert_model.ckpt.meta': 'd2ddff3ed33b80091eac95171e94149736ea74eb645e575d942ec4a5e01a40a1',
            'vocab.txt': 'eeaa9875b23b04b4c54ef759d03db9d1ba1554838f8fb26c5d96fa551df93d02',
        }
        
	# Relate SHA to urls for loop below
        self.model_sha = {
            'bert_base_uncased': self.bert_base_uncased_sha,
            'bert_large_uncased': self.bert_large_uncased_sha,
            'bert_base_cased': self.bert_base_cased_sha,
            'bert_large_cased': self.bert_large_cased_sha,
        }

    def sha256sum(self, filename):
        h  = hashlib.sha256()
        b  = bytearray(128*1024)
        mv = memoryview(b)
        with open(filename, 'rb', buffering=0) as f:
            for n in iter(lambda : f.readinto(mv), 0):
                h.update(mv[:n])

        return h.hexdigest()

    def download(self):
        for model in self.model_urls:
            url = self.model_urls[model][0]
            file = self.save_path + '/' + self.model_urls[model][1]

            print('[{}] Downloading {}'.format(self.name, url))
            if os.path.isfile(file):
                print('[{}] ** Download file already exists, skipping '
                      'download and extraction'.format(self.name))
                continue
            else:
                response = urllib.request.urlopen(url)
                with open(file, 'wb') as handle:
                    handle.write(response.read())

            print('[{}] Unzipping {}'.format(self.name, file))
            zip = zipfile.ZipFile(file, 'r')
            zip.extractall(self.save_path)
            zip.close()

            sha_dict = self.model_sha[model]
            for extracted_file in sha_dict:
                sha = sha_dict[extracted_file]
                if sha != self.sha256sum(file[:-4] + '/' + extracted_file):
                    print('[{}] SHA256sum does not match on file: {} from '
                          'download url: {}'.format(
                          self.name, extracted_file, url))
                else:
                    print('[{}] {} verified'.format(
                            self.name, file[:-4] + '/' + extracted_file))


class WikiCorpusDownloader(DatasetDownloader):
    def __init__(self, save_path):
        super(WikiCorpusDownloader, self).__init__(save_path, 'wikicorpus')

        self.download_urls = {
                'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2' : 'wikicorpus_en.xml.bz2'
        }

    def extract(self):
        for url, dst in self.download_urls.items():
            cfile = os.path.join(self.save_path, dst)
            print('[{}] Extracting: {}'.format(self.name, cfile))
            if os.path.isfile(cfile.rsplit('.', 1)[0]):
                print('[{}] ** Extracted file already exists, skipping '
                      'extraction'.format(self.name))
            else:
                subprocess.run('bzip2 -dk ' + cfile, shell=True, check=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NLP Dataset Downloader')

    parser.add_argument('--dir', type=str, required=True,
                        help='Directory to download datasets to.')
    parser.add_argument('--datasets', type=str, required=True, nargs='+',
                        choices=['wikicorpus', 'bookscorpus', 'squad',
                                 'sst-2', 'mprc', 'weights'],
                        help='Datasets to download. Each datasets will be '
                             'saved to save_path/dataset_name')
    args = parser.parse_args()

    print('Downloading {} to "{}"'.format(args.datasets, args.dir))

    downloader = Downloader(args.dir)
    for dataset in args.datasets:
        downloader.download(dataset)

    print('Finished downloading')

