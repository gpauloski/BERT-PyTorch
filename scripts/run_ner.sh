#!/bin/bash

# OPTIONS: JNLPBA, CoNLL-2003
DATASET=JNLPBA

MODEL_CONFIG=config/bert_large_uncased_config.json
MODEL_PATH=results/bert_pretraining/ckpt_8601.pt

if [ "$DATASET" == "CoNLL-2003" ]; then
    DATA_DIR=/lus/theta-fs0/projects/SuperBERT/datasets/download/ner/CoNLL-2003
    LABELS="O B-PER I-PER B-ORG I-ORG B-MISC I-MISC B-LOC I-LOC"
elif [ "$DATASET" == "JNLPBA" ]; then
    DATA_DIR=/lus/theta-fs0/projects/SuperBERT/datasets/download/ner/JNLPBA
    LABELS="O I-DNA B-DNA I-RNA B-RNA I-cell_line B-cell_line I-protein B-protein I-cell_type B-cell_type"
else
    echo "Unknown dataset $DATASET"
    exit 1
fi

# Default params from: 
# https://huggingface.co/fran-martinez/scibert_scivocab_cased_ner_jnlpba

python run_ner.py \
    --train_file $DATA_DIR/train.txt \
    --val_file $DATA_DIR/dev.txt \
    --test_file $DATA_DIR/test.txt \
    --labels $LABELS \
    --model_config_file $MODEL_CONFIG \
    --model_checkpoint $MODEL_PATH \
    --lr 5e-7 \
    --epochs 20 \
    --batch_size 32 \
    --max_seq_len 128 \
    --uppercase

