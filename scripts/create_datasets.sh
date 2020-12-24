#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo "ERROR: Missing path argument"
    echo "Usage: ./create_datasets.sh /path/for/datasets"
    exit 1
fi

DATA_PATH=${1}

N_PROCESSES=16
DOWNLOAD_PATH=$DATA_PATH/download
FORMAT_PATH=$DATA_PATH/formatted
ENCODED_PATH=$DATA_PATH/encoded
VOCAB_FILE=data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt

# Download datasets
python bert/download.py --dir $DOWNLOAD_PATH --datasets bookscorpus
python bert/download.py --dir $DOWNLOAD_PATH --datasets wikicorpus
python bert/download.py --dir $DOWNLOAD_PATH --datasets squad
python bert/download.py --dir $DOWNLOAD_PATH --datasets weights

# Extract articles from wikicorpus xml file into files of 100M each
mkdir -p $DOWNLOAD_PATH/wikicorpus/data
wikiextractor $DOWNLOAD_PATH/wikicorpus/wikicorpus_en.xml --bytes 100M \
    --processes $N_PROCESSES --output $DOWNLOAD_PATH/wikicorpus/data

# Extract the data into text files where each line is a sentence and each
# article is separated by a blank line. To prevent massive files,
# articles/books are distributed across shards.
python bert/format.py --dataset wikicorpus --processes $N_PROCESSES \
    --input_dir $DOWNLOAD_PATH/wikicorpus/data \
    --output_dir $FORMAT_PATH/wikicorpus --shards 256
python bert/format.py --dataset bookscorpus --processes $N_PROCESSES \
    --input_dir $DOWNLOAD_PATH/bookscorpus/data \
    --output_dir $FORMAT_PATH/bookscorpus --shards 256

# NOTE: at this point, the $DOWNLOAD_PATH/wikicorpus and 
#       $DOWNLOAD_PATH/bookscorpus directories can be deleted to free up space.

# Encode
#   - this is the standard BERT static masking encoder.
#   - each process can use around 10 GB of RAM so adjust $N_PROCESSES 
#     appropriately.
python bert/static_encode_pretraining_data.py \
    --input_dir $FORMAT_PATH --output_dir $ENCODED_PATH \
    --do_lower_case --max_seq_length 128 --max_predictions_per_seq 20
    --vocab_file $VOCAB_FILE --processes $N_PROCESSES
python bert/static_encode_pretraining_data.py \
    --input_dir $FORMAT_PATH --output_dir $ENCODED_PATH \
    --do_lower_case --max_seq_length 512 --max_predictions_per_seq 80
    --vocab_file $VOCAB_FILE --processes $N_PROCESSES

