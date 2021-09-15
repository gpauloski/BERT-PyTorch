#!/bin/bash

# OPTIONS: JNLPBA, CoNLL-2003
DATASET=CoNLL-2003

MODEL_CONFIG=config/bert_large_uncased_config.json
MODEL_PATH=results/bert_pretraining/ckpt_8601.pt

if [ "$DATASET" == "CoNLL-2003" ]; then
    DATA_DIR=/lus/theta-fs0/projects/SuperBERT/datasets/download/ner/CoNLL-2003
    LABELS="O B-PER I-PER B-ORG I-ORG B-MISC I-MISC B-LOC I-LOC"
    EPOCHS=5
    LR=5e-6
    UPPERCASE=true
elif [ "$DATASET" == "JNLPBA" ]; then
    DATA_DIR=/lus/theta-fs0/projects/SuperBERT/datasets/download/ner/JNLPBA
    LABELS="O I-DNA B-DNA I-RNA B-RNA I-cell_line B-cell_line I-protein B-protein I-cell_type B-cell_type"
    EPOCHS=5
    LR=5e-6
    UPPERCASE=true
elif [ "$DATASET" == "NCBI" ]; then
    DATA_DIR=/lus/theta-fs0/projects/SuperBERT/datasets/download/ner/NCBI-disease
    LABELS="O B-Disease I-Disease"
    EPOCHS=5
    LR=5e-6
    UPPERCASE=true
elif [ "$DATASET" == "BC5CDR" ]; then
    DATA_DIR=/lus/theta-fs0/projects/SuperBERT/datasets/download/ner/bc5cdr
    LABELS="O B-Entity I-Entity"
    EPOCHS=5
    LR=5e-6
    UPPERCASE=true
#elif [ "$DATASET" == "SCIIE" ]; then
#    DATA_DIR=/lus/theta-fs0/projects/SuperBERT/datasets/download/ner/sciie
#    LABELS="O I-Generic B-Generic I-Task B-Task I-Material B-Material I-Method B-Method I-Metric I-OtherScientificTerm B-OtherScientificTerm"
#    EPOCHS=5
#    LR=5e-6
#    UPPERCASE=true
else
    echo "Unknown dataset $DATASET"
    exit 1
fi

# Default params from: 
# https://huggingface.co/fran-martinez/scibert_scivocab_cased_ner_jnlpba

KWARGS=""
if [ "$UPPERCASE" == true ]; then
    KWARGS=" --uppercase "
fi

python run_ner.py \
    --train_file $DATA_DIR/train.txt \
    --val_file $DATA_DIR/dev.txt \
    --test_file $DATA_DIR/test.txt \
    --labels $LABELS \
    --model_config_file $MODEL_CONFIG \
    --model_checkpoint $MODEL_PATH \
    --lr $LR \
    --epochs $EPOCHS \
    --batch_size 32 \
    --max_seq_len 128 \
    $KWARGS

