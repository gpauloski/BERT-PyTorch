#!/bin/bash

#DATA_DIR=/lus/theta-fs0/projects/SuperBERT/datasets/download/ner/JNLPBA
DATA_DIR=/lus/theta-fs0/projects/SuperBERT/datasets/download/ner/CoNLL-2003
VOCAB_FILE=/lus/theta-fs0/projects/SuperBERT/datasets/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt
#LABELS="O I-DNA B-DNA I-RNA B-RNA I-cell_line B-cell_line I-protein B-protein I-cell_type B-cell_type"
LABELS="O B-PER I-PER B-ORG I-ORG B-MISC I-MISC B-LOC I-LOC"
MODEL_CONFIG=config/bert_large_config.json
MODEL_PATH=results/bert_pretraining/ckpt_8601.pt

# PARAMS: https://huggingface.co/fran-martinez/scibert_scivocab_cased_ner_jnlpba
python run_ner.py \
    --train_file $DATA_DIR/train.txt \
    --val_file $DATA_DIR/dev.txt \
    --test_file $DATA_DIR/test.txt \
    --vocab_file $VOCAB_FILE \
    --labels $LABELS \
    --model_config_file $MODEL_CONFIG \
    --model_checkpoint $MODEL_PATH \
    --epochs 6 \
    --lr 5e-5 \
    --batch_size 32 \
    --max_seq_len 128 \
    #--uppercase

