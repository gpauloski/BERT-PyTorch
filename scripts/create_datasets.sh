#!/bin/bash

DATA_PATH="data"
INCLUDE_BOOKS=false
N_PROCESSES=4

while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | sed 's/^[^=]*=//g'`
    if [[ "$VALUE" == "$PARAM" ]]; then
        shift
        VALUE=$1
    fi
    case $PARAM in
        -h|--help)
            echo "USAGE: ./scripts/create_datasets.sh"
            echo "  -h,--help         Display this help message"
            echo "  -d,--dir   [int]  Output directory (default: data/)"
            echo "  -p,--nproc [int]  Number of processes (default: 4)"
            echo "  -b,--books        Download Bookscorpus (default: false)"
            exit 0
        ;;
        -d|--dir)
            DATA_PATH=$VALUE
        ;;
        -p|--nproc)
            N_PROCESSES=$VALUE
        ;;
        -b|--books)
            INCLUDE_BOOKS=true
        ;;
        *)
          echo "ERROR: unknown parameter \"$PARAM\""
          exit 1
        ;;
    esac
    shift
done


DOWNLOAD_PATH=$DATA_PATH/download
FORMAT_PATH=$DATA_PATH/formatted
ENCODED_PATH=$DATA_PATH/encoded
VOCAB_FILE=data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt

# Download datasets
#python bert/download.py --dir $DOWNLOAD_PATH --datasets wikicorpus
#python bert/download.py --dir $DOWNLOAD_PATH --datasets squad
#python bert/download.py --dir $DOWNLOAD_PATH --datasets weights
#if [ "$INCLUDE_BOOKS" == true ]; then
#    python bert/download.py --dir $DOWNLOAD_PATH --datasets bookscorpus
#fi

# Extract articles from wikicorpus xml file into files of 100M each
#mkdir -p $DOWNLOAD_PATH/wikicorpus/data
#wikiextractor $DOWNLOAD_PATH/wikicorpus/wikicorpus_en.xml --bytes 100M \
#    --processes $N_PROCESSES --output $DOWNLOAD_PATH/wikicorpus/data

# Extract the data into text files where each line is a sentence and each
# article is separated by a blank line. To prevent massive files,
# articles/books are distributed across shards.
#python bert/format.py --dataset wikicorpus --processes $N_PROCESSES \
#    --input_dir $DOWNLOAD_PATH/wikicorpus/data \
#    --output_dir $FORMAT_PATH/wikicorpus --shards 256
#if [ "$INCLUDE_BOOKS" == true ]; then
#    python bert/format.py --dataset bookscorpus --processes $N_PROCESSES \
#        --input_dir $DOWNLOAD_PATH/bookscorpus/data \
#        --output_dir $FORMAT_PATH/bookscorpus --shards 256
#fi

# NOTE: at this point, the $DOWNLOAD_PATH/wikicorpus and 
#       $DOWNLOAD_PATH/bookscorpus directories can be deleted to free up space.

# Encode
#   - this is the standard BERT static masking encoder.
#   - each process can use around 10 GB of RAM so adjust $N_PROCESSES 
#     appropriately.
python bert/static_encode_pretraining_data.py \
    --input_dir $FORMAT_PATH --output_dir $ENCODED_PATH \
    --do_lower_case --max_seq_length 128 --max_predictions_per_seq 20 \
    --vocab_file $VOCAB_FILE --processes $N_PROCESSES
python bert/static_encode_pretraining_data.py \
    --input_dir $FORMAT_PATH --output_dir $ENCODED_PATH \
    --do_lower_case --max_seq_length 512 --max_predictions_per_seq 80 \
    --vocab_file $VOCAB_FILE --processes $N_PROCESSES

