# BERT For PyTorch

This repository provides scripts for pretraining and finetuning BERT in PyTorch.

 
## Table of Contents

- [Overview](#overview)
- [Quickstart Guide](#quickstart-guide)
- [ThetaGPU Quickstart](#thetagpu-quickstart)
- [Pretraining Methods](#pretraining-methods)
  * [BERT (two phase, static masking)](#standard-bert-pretraining)
  * [RoBERTa (single phase, dynamic masking)](#roberta-pretraining)
- [Performance](#performance)
  * [Pretraining](#pretraining)
  * [Finetuning](#finetuning)


## Overview

This repository provides scripts for data downloading, preprocessing, pretraining and finetuning [BERT](https://arxiv.org/abs/1810.04805) (Bidirectional Encoder Representations from Transformers). 
This implementation is based on the [NVIDIA implementation of BERT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT) which is an optimized version of the [Hugging Face](https://huggingface.co/) and [Google](https://github.com/google-research/bert) implementations.

The major differences between the NVIDIA and original (Hugging Face) implementations are [[ref](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#model-overview)]:
- Data downloading and preprocesssing scripts
- Fused [LAMB](https://arxiv.org/pdf/1904.00962.pdf) optimizer for large batch training
- Fused Adam optimizer for fine tuning
- Fused CUDA kernels for LayerNorm
- Automatic mixed precision training with [NVIDIA Apex](https://github.com/NVIDIA/apex)
- Benchmarking scripts for Docker containers

The major differences between this version and the NVIDIA implementation are:
- Scripts are designed to run in the included Conda environment rather than a Docker container
- Enhanced data preprocessing to support multiprocessing where possible
- [PyTorch AMP](https://pytorch.org/docs/stable/amp.html) instead of Apex
- Quality of life changes
  - Better logging with TensorBoard support
  - Pretraining config files
  - Code refactoring (remove unused code, file structure changes, etc.)
- [K-FAC](https://github.com/gpauloski/kfac_pytorch) support
- Scripts for launching training in distributed environments (multi-gpu, multi-node, SLURM and Cobalt clusters)
- [RoBERTa](https://arxiv.org/abs/1907.11692) optimizations (dynamic masking)


## Quickstart Guide

### **1. Create Conda environment**

```
$ conda create -n bert-pytorch python=3.8
$ conda activate bert-pytorch
$ conda env update --name bert-pytorch --file environment.yml
$ pip install -r requirements.txt
```

Install NVIDIA APEX. Note this step requires `nvcc` and may fail if done on systems without a GPU (i.e. you may need to install on a compute node).
```
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### **2. Build datasets** 
Skip this section if you already have the encoded dataset.

Build pretraining dataset using Wikipedia and BooksCorpus and save to `data/`.
```
$ ./scripts/create_datasets.sh --output data --nproc 8 --download --format --encode
```
See `./scripts/create_datasets.sh --help` for all options.
Downloading can take many hours and the BooksCorpus servers can easily be overloaded when download from scratch so to skip the BooksCorpus, include the `--no-books` flag.
Formatting and encoding can be made faster by increasing the number of processes used in the script at the cost of more RAM.
This steps will use a couple hundred GBs of disk space; however the `download` directory and `formatted` directory can be deleted if you do not plan to rebuild the dataset again.

### **3. Training**
Arguments to `run_pretraining.py` can be passed as command line arguments or as key-value pairs in the config files.
Command line arguments take precendence over config file arguments.
See `python run_pretraining.py --help` for a full list of arguments and their defaults.

#### Single-Node Multi-GPU Training
```
$ python -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE run_pretraining.py --config_file $PHASE1_CONFIG --input_dir $PHASE1_DATA --output_dir $OUTPUT_DIR
```
An example `$PHASE1_CONFIG` is provided in `config/bert_pretraining_phase1_config.json`.
The example configs are tuned for 40GB NVIDIA A100s.
The training script will recursively find and use all HDF5 files in `$PHASE1_DATA`.
Training logs and checkpoints are written to `$OUTPUT_DIR`.
After phase 1 training is finished, continue with phase 2 by running the same command with the phase 2 config and data paths (the output directory stays the same).
   
Training logs are written to `$OUTPUT_DIR/log.txt`, and TensorBoard can be used for monitoring with `tensorboard --logdir=$OUTPUT_DIR`.

#### Multi-Node Multi-GPU Training
```
$ python -m torch.distributed.launch --node_rank=$RANK --master_addr=$MASTER --nnodes=$NODE_COUNT --nproc_per_node=$GPUS_PER_NODE run_pretraining.py --config $PHASE1_CONFIG --input_dir $PHASE1_DATA --output_dir $OUTPUT_DIR
```
This command must be run on each node where the `--node_rank` is changed appropriately.
`$MASTER` is the hostname of the first node (e.g. thetagpu08).

#### Automatic Multi-Node Training
Example scripts for running pretraining on SLURM or Cobalt clusters are provided in `scripts/run_pretraining.{sbatch,cobalt}`.
The scripts will automatically infer the distributed training configuration from the nodelist and launch the PyTorch distributed processes.
The paths and environment setups are examples so you will need to update the scripts for your specific needs.


## ThetaGPU Quickstart

For members of this project only (you will not have access to the premade dataset otherwise)! 

### 1. Build conda environment

```
$ source /lus/theta-fs0/software/thetagpu/conda/pt_master/2020-11-25/mconda3/setup.sh
$ conda create -n bert-pytorch python=3.8
$ conda activate bert-pytorch
$ conda env update --name bert-pytorch --file environment.yml
$ pip install -r requirements.txt
```

Install NVIDIA APEX. Note this step requires `nvcc` and may fail if done on systems without a GPU.
Start an interactive session on a compute node to install APEX: `qsub -A $PROJECT_NAME -I -q full-node -n $NODE_COUNT -t $TIME`.
```
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### 2. Build dataset
The provided phase 1 and 2 config files point to premade datasets that are already on Theta so this step can be skipped.
If you would like to build the dataset again, you can skip the download step (the script will automatically use the downloaded data on Theta).
Dataset building can be done on a login node or Theta compute node (not ThetaGPU node because the GPUs will not be used).
```
$ ./scripts/create_datasets.sh --output data --nproc 16 --format --encode
```

### 3. Training
To launch training on a single node in an interactive qsub session using the existing dataset:
```
$ export OUTPUT_DIR=results/bert_pretraining
$ export PHASE1_CONFIG=config/bert_pretraining_phase1_config.json
$ export PHASE2_CONFIG=config/bert_pretraining_phase2_config.json
$ export PHASE1_DATA=/lus/theta-fs0/projects/SuperBERT/datasets/encoded/bert_masked_wikicorpus_en/phase1
$ export PHASE2_DATA=/lus/theta-fs0/projects/SuperBERT/datasets/encoded/bert_masked_wikicorpus_en/phase2
$ # PHASE 1
$ python -m torch.distributed.launch --nproc_per_node=8 run_pretraining.py --config_file $PHASE1_CONFIG --input_dir $PHASE1_DATA --output_dir $OUTPUT_DIR
$ # PHASE 2
$ python -m torch.distributed.launch --nproc_per_node=8 run_pretraining.py --config_file $PHASE2_CONFIG --input_dir $PHASE2_DATA --output_dir $OUTPUT_DIR
```

For automatic multi-node training in an interactive qsub session:
```
$ ./scripts/run_pretraining.cobalt
```
The training environment will be automatically configured using the `$COBALT_NODEFILE`.
Config files and data paths can be modified at the top of the cobalt script.
To run the default phase 2 training, set `PHASE=2` in the script.

This script can also be submitted as a Cobalt job:
```
$ qsub scripts/run_pretraining.cobalt
```
Modify the Cobalt job specifications at the top of the script as needed.

To monitor training with TensorBoard, see the [Theta TensorBoard Instructions](https://www.alcf.anl.gov/support-center/theta/tensorboard-instructions).


## Pretraining Methods

### Standard BERT Pretraining

### RoBERTa Pretraining


## Performance

### Pretraining

### Finetuning

## FAQ

### **free(): invalid pointer on Theta**

Generally this is related to PyTorch being compiled without MAGMA.
Try installing PyTorch via pip instead of Conda.
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

## TODO

- [ ] Benchmarking performance for README
- [ ] Add option to limit maximum file size for `format.py` instead of choosing `n_shard`.
- [ ] Go through TODOs in repo
- [x] Add byte-pair encoding option
- [x] specify log file prefix as argument
- [ ] whole word masking in encode data
