import h5py
import numpy as np
import os
import threading
import torch
import warnings


class ShardedPretrainingDataset(torch.utils.data.Dataset):
    """BERT Pretraining Dataset with Dynamic Masking

    BERT Pretraining dataset that supports dynamic masking and multiple input
    files. In particular, the dataset is designed to load at most two input
    files into memory at a time. The first file is the file samples are drawn
    from, and the second file is loaded in a background to be ready for when
    all samples from the first file are exhausted.

    TODO:
      - Shuffling (we leave shuffling to the dataset creating for now).
        Shuffling poses a slight problem because we can only shuffle indices
        within the current file (otherwise the sample index would jump across
        files) and multiple ranks could be working on a separate chunk of the
        same file. Thus, we would have to only shuffle within the open file
        and subset of indices assigned to this rank which would take
        coordination with the Distributed Sampler.
      - If torch distributed is initialized and world size > 1, just scan
        input files on rank 0 and broadcast file start/end indices to all other
        ranks to prevent every rank from opening files at same time

    Notes:
      - __getitem__(idx) should be called with sequential values for idx. This
        is because we want to avoid constantly loading new files into memory.
        By using sequential values for idx, we can exhaust all samples in a file
        that is loaded into memory before moving onto the next file.
      - For distributed training, use the included DistributedSampler. The 
        standard PyTorch DistributedSampler does a round-robin partition of
        indices which would violate calling __getitem__ with sequential index
        values. The included DistributedSampler chunks the indices between
        ranks so that the Sampler returns sequential indices and that different
        workers start on different files.
      - The first call to __getitem__ will be longer than the average call
        because the Dataset object does not know which file we are starting with
        so the file must be inferred from the starting idx and loaded.
      - For distributed training, calling set_epoch at the start of each epoch
        is necessary for having different shuffled indices each epoch. If using
        the DistributedSampler, DistributedSampler.set_epoch() will call this
        class's set_epoch.

    Input File Format:
      HDF5 file with keys = ['input_ids', 'special_token_positions', 
      'next_sentence_labels']. The value for each key is a list with length 
      num_samples. Each item in input_ids is a list[int] of encoded tokens 
      padded to max_seq_len with zeros. Each item in special_token_positions
      is a list[int] of position in the corresponding item in inputs_ids
      where either a [CLS] or [SEP] token is. The special token positions are
      used to ensure that a [CLS] or [SEP] token is not masked. Each item
      if next_sentence_labels is 1 if there are two sequences in input_ids and
      the second sequence is a random next sequence (i.e. not the actual
      next sequence that followed in the document) else 0.

    Args:
      files (list[str]): list of paths to input files
      mask_token_index (int): Integer value for the '[MASK]' token to replace
          masked values with in the input.
      max_pred_per_seq (int): Maximum number of masked tokens per sequence
      masked_lm_prob (float): Fraction of tokens in input to mask
      vocab_size (int): Number of tokens in vocab. Used to sample random
          tokens when masking
      original_token_prob (float, optional): Probability to keep original token
          rather than masking (default: 0.1)
      random_token_prob (float, optional): Probability to replace with random
          token rather than masking (default: 0.1)
      shuffle (bool, optional): Not supported yet (default=False)
      seed (int, optional): Seed for shuffling and random masking (default=None)
    """
    def __init__(self,
                 files,
                 mask_token_index,
                 max_pred_per_seq,
                 masked_lm_prob,
                 vocab_size,
                 original_token_prob=0.1,
                 random_token_prob=0.1,
                 shuffle=False,
                 seed=None):

        if not isinstance(mask_token_index, int) and mask_token_index is not None:
            raise ValueError('mask_token_index must be an integer')
        if not isinstance(max_pred_per_seq, int) or max_pred_per_seq < 0:
            raise ValueError('max_pred_per_seq must be an integer >= 0')
        if masked_lm_prob < 0 or masked_lm_prob > 1:
            raise ValueError('masked_lm_prob must be in [0,1]')
        if not isinstance(vocab_size, int) or vocab_size < 0:
            raise ValueError('vocab_size must be an integer >= 0')
        if original_token_prob < 0 or original_token_prob > 1:
            raise ValueError('original_token_prob must be in [0,1]')
        if random_token_prob < 0 or random_token_prob > 1:
            raise ValueError('random_token_prob must be in [0,1]')
        if random_token_prob + original_token_prob > 1:
            raise ValueError('random_token_prob + original_token_prob > 1')
        if shuffle:
            raise ValueError('Shuffling the dataset is not supported yet.'
                             'It is recommended to pre shuffle the samples in '
                             'the input files.')

        if isinstance(files, str):
            files = [files]
        files.sort()  # Ensure processes see files in same order
        self.files, self.file_idxs = self._verify_and_count_samples(files)

        self.mask_token_index = mask_token_index
        self.max_pred_per_seq = max_pred_per_seq
        self.masked_lm_prob = masked_lm_prob
        self.vocab_size = vocab_size
        self.original_token_prob = original_token_prob
        self.random_token_prob = random_token_prob

        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        if self.seed is not None:
            np.random.seed(self.seed)

        # We use rank to determine which file should be loaded next so different
        # rank open different files
        self.file_idx = None
        self.next_file_idx = None
        self.file_sample_start_idx = -1
        # force the idx in the first call to __getitem__(idx) to be greater
        # than this value
        self.file_sample_end_idx = -1
        self.data = None  # Data for current file

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.file_idxs[-1][1]

    def __getitem__(self, idx):
        if self.data is None:
            # We only enter this once: the first time __getitem__ is called
            # for this instance of the Dataset
            self.next_file_idx = self._get_file_idx_from_sample_idx(idx)
            self.next_file_thread = self._async_load_file(self.next_file_idx)

        if idx >= self.file_sample_end_idx or idx < self.file_sample_start_idx:
            # Done with data for current file so get next data from future
            del self.data  # force clear this memory
            self.next_file_thread.join()
            self.data = self.next_file_data
            self.file_idx = self.next_file_idx
            self.next_file_idx += 1
            if self.next_file_idx == len(self.files):
                self.next_file_idx = 0
            self.next_file_thread = self._async_load_file(self.next_file_idx)
            self.file_sample_start_idx = self.file_idxs[self.file_idx][0]
            self.file_sample_end_idx = self.file_idxs[self.file_idx][1]

        if idx >= self.file_sample_end_idx or idx < self.file_sample_start_idx:
            # We check the same conditions again since we may have loaded a new
            # file and want to make sure the correct file was loaded
            raise RuntimeError('idx ({}) out of range ({}, {}) for current '
                               'file. This can happen when calling __getitem__'
                               ' with out of order indices (e.g. when using a '
                               'sampler with shuffle=True).'.format(
                               idx, self.file_sample_start_idx,
                               self.file_sample_end_idx))

        idx -= self.file_sample_end_idx
        input_ids = self.data['input_ids'][idx]
        next_sentence_label = self.data['next_sentence_labels'][idx]

        if 'special_token_positions' in self.data:
            special_token_positions = self.data['special_token_positions'][idx]
            segment_ids = self._get_segment_ids(
                    input_ids, special_token_positions)
            input_mask = self._get_input_mask(
                    input_ids, special_token_positions)
            masked_input_ids, masked_lm_labels = self._mask_input(
                    input_ids, special_token_positions)
        else:
            # Legacy support for premasked dataset using format from 
            # NVIDIA/DeepLearningExamples/PyTorch/LanguageModeling/BERT
            segment_ids = self.data['segment_ids'][idx]
            input_mask = self.data['input_mask'][idx]
            masked_lm_positions = self.data['masked_lm_positions'][idx]
            masked_lm_ids = self.data['masked_lm_ids'][idx]
            masked_input_ids = input_ids
            masked_lm_labels = self._get_masked_labels(
                    input_ids, masked_lm_positions, masked_lm_ids)

        return [
            masked_input_ids.astype(np.int64),
            segment_ids.astype(np.int64),
            input_mask.astype(np.int64),
            masked_lm_labels.astype(np.int64),
            np.asarray(next_sentence_label).astype(np.int64)
        ]
        
    def _get_file_idx_from_sample_idx(self, idx):
        """Get file idx containing the sample idx"""
        for i, (start_idx, end_idx) in enumerate(self.file_idxs):
            if start_idx <= idx < end_idx:
                return i
        raise ValueError('idx ({}) exceeds dataset size ({})'.format(
                idx, self.__len__()))

    def _async_load_file(self, file_idx):
        """Returns handle for thread that will load file in background"""
        th = threading.Thread(target=self._get_dict_from_hdf5, 
                args=(self.files[file_idx],))
        th.start()
        return th

    def _get_dict_from_hdf5(self, filepath):
        """Load HDF5 file and return dict of numpy arrays"""
        self.next_file_data = {}
        with h5py.File(filepath, 'r') as f:
            for key in f.keys():
                self.next_file_data[key] = np.asarray(f[key][:])

    def _get_segment_ids(self, input_ids, special_token_positions):
        """Get segment ids list

        Examples:
          Input:  [CLS] seq tokens [SEP] padding
          Output:   0     0 ... 0    0   0 ... 0

          Input:  [CLS] seq tokens [SEP] seq tokens [SEP] padding
          Output:   0     0 ... 0    0     1 ... 1    1   0 ... 0
        """
        segment_ids = np.zeros_like(input_ids)
        if len(special_token_positions) == 3:
            segment_ids[special_token_positions[1] + 1: 
                    special_token_positions[2] + 1] = 1
        return segment_ids
    
    def _get_input_mask(self, input_ids, special_token_positions):
        """Get mask of input (to ignore padding)

        Examples:
          Input:  [CLS] seq tokens [SEP] padding
          Output:   1     1 ... 1    1   0 ... 0

          Input:  [CLS] seq tokens [SEP] seq tokens [SEP] padding
          Output:   1     1 ... 1    1     1 ... 1    1   0 ... 0
        """
        input_mask = np.zeros_like(input_ids)
        input_mask[:special_token_positions[-1] + 1] = 1
        return input_mask
    
    def _get_masked_labels(self, input_ids, masked_lm_positions, masked_lm_ids):
        """Get masked labels

        Args:
          input_ids (array)
          masked_lm_position (array): index in input_ids of masked tokens
          masked_lm_ids (array): true token value for each position in
              masked_lm_position.

        Returns:
          Array with length == len(input_ids) with the true value for
          each correspoinding masked token in input_ids and -1 for all
          tokens in input_ids which are not masked.
        """
        masked_lm_labels = np.ones_like(input_ids) * -1
        index = len(input_ids)
        # store number of  masked tokens in index
        padded_mask_indices = np.nonzero(masked_lm_positions == 0)
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0][0]
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]
        return masked_lm_labels

    def _mask_input(self, input_ids, special_token_positions):
        """Randomly mask the input"""
        masked_lm_labels = np.ones_like(input_ids) * -1
        # note we get indices up to special_token_positions[-1] b/c
        # everything after the last special token is padding
        indices = [i for i in range(special_token_positions[-1]) 
                if i not in special_token_positions]
        mask_count = min(self.max_pred_per_seq, 
                         max(1, int(len(indices) * self.masked_lm_prob)))
        mask_indices = np.random.choice(indices, mask_count)
        masked_lm_labels[mask_indices] = input_ids[mask_indices]
        for idx in mask_indices:
            rng = np.random.rand()
            if rng < self.original_token_prob:
                continue
            elif rng < self.original_token_prob + self.random_token_prob:
                input_ids[idx] = np.random.randint(0, self.vocab_size - 1)
            else:
                input_ids[idx] = self.mask_token_index
        return input_ids, masked_lm_labels

    def _verify_and_count_samples(self, files):
        """Check files can be opened, contain correct keys, and count samples
        
        Args:
            files (list, str): list of files paths
            
        Returns:
            list[filepaths], list[Tuple(file_start_idx, file_end_idx)]
            Note file_end_idx is not inclusive
        """
        current_idx = 0
        verified_files = []
        verified_file_idxs = []
        # Note: there are more keys than just these two in the input files but
        # we support two kinds of inputs (masked and unmasked) and the
        # remaining keys are different between the files
        keys = ['input_ids', 'next_sentence_labels']
        for fpath in files:
            if not os.path.isfile(fpath):
                warnings.warn('File not found: {}. Skipping File'.format(fpath))
                continue
            try:
                counts = []
                with h5py.File(fpath, 'r') as f:
                    for key in keys:
                        counts.append(len(f[key]))
            except:
                warnings.warn('Unable to read keys ({}) from {}. '
                              'Skipping File'.format(keys, fpath))
                continue
            if len(set(counts)) != 1:
                warnings.warn('Number of samples per key in {} '
                              'do not match. Skipping File'.format(fpath))
                continue
            verified_files.append(fpath)
            last_idx = current_idx + counts[0]
            verified_file_idxs.append((current_idx, last_idx))
            current_idx = last_idx
        if len(files) == 0:
            raise RuntimeError('Unable to open any valid data files')
        return verified_files, verified_file_idxs


class DistributedSampler(torch.utils.data.distributed.DistributedSampler):
    """Custom Distributed Sampler for ShardedPretrainingDataset
    
    The PyTorch DistributedSampler partitions indices in the dataset
    round-robin across the workers. For a sharded dataset where sample indices
    are divided across files that are swapped in and out of memory, this
    will mean all workers access the same files at the same time. This custom
    DistributedSampler chunks the indices across workers to minimize the
    number of workers accessing the same file and the number of file swaps
    required.

    Because we sample indices sequentially, we leave the shuffling to the
    dataset so that shuffling samples is done among the samples for the open
    file rather globally for all samples in all files.

    Note: Standard PyTorch samplers create and return an iteration whereas
    this class is itself an iteration. This allows us to save the sampler
    to the state_dict and load state_dict to resume training from the same
    sample.
    """
    def __init__(self, *args, **kwargs):
        kwargs['shuffle'] = False
        super(DistributedSampler, self).__init__(*args, **kwargs)
        self.dataset.seed = self.seed
        
        # This sampler is deterministic so we can get list of indices once
        indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * 
                        math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        self.global_indices = indices
        self.index = 0

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == self.num_samples:
            self.index = 0
            raise StopIteration()
        else:
            # Add offset for current rank
            x = self.global_indices[self.index + self.rank * self.num_samples]
            self.index += 1
            return x

    def state_dict(self):
        return {
            'epoch': self.epoch,
            'seed': self.seed,
            'num_replicas': self.num_replicas,
            'total_size': self.total_size,
            'index': self.index
        }

    def load_state_dict(self, state_dict):
        if state_dict['total_size'] != self.total_size:
            warnings.warn('The number of samples in the Sampler has changed. '
                          'Skipping restoring sampler state. If the dataset '
                          'was changed and the sampler should be reset, '
                          'ignore this message')
            return
        if state_dict['num_replicas'] != self.num_replicas:
            warnings.warn('The number of replicas has changed so the resume '
                          'index from the sampler is no longer valid. '
                          'Skipping restoring sampler state.')
            return
        self.epoch = state_dict['epoch']
        self.seed = state_dict['seed']
        self.index = state_dict['index']

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)


if __name__ == '__main__':
    import argparse
    import tqdm

    from pathlib import Path

    parser = argparse.ArgumentParser(description='Dataloader test')
    parser.add_argument('--input_dir', default=None, type=str,
                        help='The input data dir containing .hdf5 files '
                             'for the task.')
    parser.add_argument('--max_predictions_per_seq', default=80, type=int,
                        help='The maximum total of masked tokens in input '
                             'sequence')
    parser.add_argument('--masked_lm_prob', type=float, default=0.15,
                        help='Specify the probability for masked lm')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Per worker batch size')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Per worker batch size')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank of worker')

    args = parser.parse_args()

    torch.distributed.init_process_group(backend='gloo', init_method='env://')
    rank = torch.distributed.get_rank()

    input_files = []
    if os.path.isfile(args.input_dir):
        input_files.append(args.input_dir)
    elif os.path.isdir(args.input_dir):
        for path in Path(args.input_dir).rglob('*.hdf5'):
            if path.is_file():
                input_files.append(str(path))

    if rank == 0:
        print('[rank {}] Found {} input files'.format(rank, len(input_files)))

    dataset = ShardedPretrainingDataset(input_files, -99, 
            args.max_predictions_per_seq, args.masked_lm_prob, vocab_size=30000)
    sampler = DistributedSampler(
            dataset, num_replicas=torch.distributed.get_world_size(), 
            rank=torch.distributed.get_rank())
    loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, sampler=sampler,
            num_workers=4, pin_memory=True)
    
    if rank == 0:
        print('[rank {}] Dataset size = {}'.format(rank, len(dataset)))
        print('[rank {}] Dataloader size = {}'.format(rank, len(loader)))
        print('[rank {}] Sampler num_samples = {}'.format(rank, sampler.num_samples))
        print('[rank {}] Sampler total_size = {}'.format(rank, sampler.total_size))
        print('[rank {}] Sampler shuffle = {}'.format(rank, sampler.shuffle))
        #print('[rank {}] Dataset files = {}'.format(rank, dataset.files))
        #print('[rank {}] Dataset file idxs = {}'.format(rank, dataset.file_idxs))

    count = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for batch in tqdm.tqdm(loader, disable=rank != 0):
            #if count % 200 == 0:
            #    print(rank, dataset.file_idx)
            #count += 1
            continue

            # print individual sample from batch
            count += 1
            if count > 10:
                break
            input_ids, segment_ids, input_mask, lm_labels, ns = batch
            input_ids, segment_ids, input_mask, lm_labels = \
                    input_ids[0], segment_ids[0], input_mask[0], lm_labels[0]
            for s in zip(input_ids, segment_ids, input_mask, lm_labels):
                print([x.item() for x in s])
            print('\n\n')

