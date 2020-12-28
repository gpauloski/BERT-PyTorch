import hdf5
import numpy as np
import os
import torch

from concurrent import futures

# DATALOADER SPEC
#   - read from multple input files
#   - swap input files in and out of memory
#   - different workers read differnet input files
#   - mask on the fly
#   - save_state_dict
#   - work with DistributedSample(shuffle=False)

class ShardedPretrainingDataset(torch.utils.data.Dataset):

    def __init__(self, files, rank=0):
        if isinstance(files, str):
            files = [files]
        files.sort()  # Ensure processes see files in same order
        self.files, self.file_idxs = self._verify_and_count_samples(self.files)

        self.pool = futures.ProcessPoolExecutor(1)

        # We use rank to determine which file should be loaded next so different
        # rank open different files
        self.file_idx = None
        self.next_file_idx = None
        self.file_sample_start_idx = -1
        # force the idx in the first call to __getitem__(idx) to be greater
        # than this value
        self.file_sample_end_idx = -1
        self.data = None  # Data for current file
        self.previous_idx = None  # used to ensure __getitem__ called in order

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.data is None:
            # We only enter this once: the first time __getitem__ is called
            # for this instance of the Dataset
            self.file_idx = self._get_file_idx_from_sample_idx(idx)
            self.next_file_future = self._async_load_file(self.file_idx)

        if previous_idx is not None and self.previous_idx + 1 != idx:
            raise RuntimeError('__getitem__(idx) must be called with '
                               'sequential indicies')
        self.previous_idx = idx

        if idx >= self.file_sample_end_idx or self.data is None:
            # Done with data for current file so get next data from future
            del self.data  # force clear this memory
            self.data = self.next_file_future.result(timeout=None)
            self.file_idx = self.next_file_idx
            self.next_file_idx += 1
            if self.next_file_idx == len(files):
                self.next_file_idx = 0
            self.next_file_future = self._async_load_file(self.next_file_idx)
            self.file_sample_start_idx = self.file_idxs[self.file_idx][0]
            self.file_sample_end_idx = self.file_idxs[self.file_idx][1]

        if idx >= self.file_sample_end_idx or idx < self.file_sample_start_idx:
            raise RuntimeError('idx ({}) out of range ({}, {}) for current '
                               'file'.format(idx, self.file_sample_start_idx,
                                             self.file_sample_end_idx))

        idx -= self.file_sample_end_idx
        input_ids = self.data['input_ids'][idx]
        special_token_positions = self.data['special_token_positions'][idx]
        next_sentence_label = self.data['next_sentence_labels'][idx]

        # TODO(gpauloski): make these methods
        segment_ids = self._get_segment_ids(input_ids, special_token_positions)
        input_mask, masked_lm_labels = self._get_mask(...)

        return [input_ids, segment_ids, input_mask, masked_lm_labels, 
                next_sentence_label]
        
    def _get_file_idx_from_sample_idx(self, idx):
        """Get file idx containing the sample idx"""
        for i, (start_idx, end_idx) in enumerate(self.file_idxs):
            if start_idx <= idx < end_idx:
                return i
        raise ValueError('idx ({}) exceeds dataset size ({})'.format(
                idx, self.__len__()))

    def _async_load_file(self, file_idx):
        """Returns future for process that will load file in background"""
        return pool.submit(self._dict_from_hdf5, self.files[file_idx])

    def _get_dict_from_hdf5(self, file):
        """Load HDF5 file and return dict of numpy arrays"""
        data = {}
        with h5py.File(file, 'r') as f:
            for key in f.keys():
                data[key] = np.asarray(f[key][:])
        return data

    def _get_mask(input_ids, special_token_positions):
        pass

    def _verify_and_count_samples(self, files):
        """Check files can be opened, contain correct keys, and count samples
        
        Args:
            files (list, str): list of files paths
            
        Returns:
            list[filepaths], list[Tuple(file_start_idx, file_end_idx)]
            Note file_end_idx is not inclusive
        """
        current_idx = 0
        files = []
        file_idxs = []
        keys = ['input_ids', 'special_token_positions', 'next_sentence_labels']
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
            files.append(fpath)
            last_idx = current_idx + counts[0]
            file_idxs.append((sample_start, last_idx))
            current_idx = last_idx
        if len(files) == 0:
            raise RuntimeError('Unable to open any valid data files')
        return files, files_idxs

