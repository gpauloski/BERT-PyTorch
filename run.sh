#!/bin/bash
#COBALT -A AIElectrolytes --attrs enable_ssh=1

source /lus/theta-fs0/software/thetagpu/conda/pt_master/2020-11-25/mconda3/setup.sh
conda activate bert-pytorch

python utils/build_vocab.py --input data/formatted/shadow_eng_10B/ --output data/vocabs/shadow_eng_10B.txt --size 50000 --uppercase
