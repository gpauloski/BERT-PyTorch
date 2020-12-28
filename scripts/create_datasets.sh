#!/bin/bash

DATA_PATH="data"
INCLUDE_BOOKS=true
N_PROCESSES=4
DOWNLOAD=false
FORMAT=false
ENCODE=false

while [[ "$1" == -* ]]; do
    case "$1" in
        -h|--help)
            echo "USAGE: ./scripts/create_datasets.sh"
            echo "  -h,--help           Display this help message"
            echo "  -o,--output [path]  Output directory (default: data/)"
            echo "  -p,--nproc  [int]   Number of processes (default: 4)"
            echo "  --no-books          Skip Bookscorpus (default: false)"
            echo "  --download          Download datasets (default: false)"
            echo "  --format            Format datasets (default: false)"
            echo "  --encode            Encode datasets (default: false)"
            exit 0
        ;;
        -o|--output)
            shift
            DATA_PATH="$1"
        ;;
        -p|--nproc)
            shift
            N_PROCESSES="$1"
        ;;
        --no-books)
            INCLUDE_BOOKS=false
        ;;
        --download)
            DOWNLOAD=true
        ;;
        --format)
            FORMAT=true
        ;;
        --encode)
            ENCODE=true
        ;;
        *)
          echo "ERROR: unknown parameter \"$1\""
          exit 1
        ;;
    esac
    shift
done


if [ "$DOWNLOAD" == true ]; then
    DOWNLOAD_PATH=$DATA_PATH/download
else
    # Path to predownloaded datasets on Theta
    DOWNLOAD_PATH=/lus/theta-fs0/projects/SuperBERT/datasets/download 
fi

FORMAT_PATH=$DATA_PATH/formatted
ENCODED_PATH=$DATA_PATH/encoded
VOCAB_FILE=$DOWNLOAD_PATH/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt

echo "./create_datasets.sh:"
echo "    download=$DOWNLOAD"
echo "    format=$FORMAT"
echo "    encode=$ENCODE"
echo "    download_path=$DOWNLOAD_PATH"
echo "    format_path=$FORMAT_PATH"
echo "    encode_path=$ENCODED_PATH"
echo "    include_books=$INCLUDE_BOOKS"
echo "    num_processes=$N_PROCESSES"


if [ "$DOWNLOAD" == true ]; then
    python bert/download.py --dir $DOWNLOAD_PATH --datasets wikicorpus
    python bert/download.py --dir $DOWNLOAD_PATH --datasets squad
    python bert/download.py --dir $DOWNLOAD_PATH --datasets weights
    if [ "$INCLUDE_BOOKS" == true ]; then
        python bert/download.py --dir $DOWNLOAD_PATH --datasets bookscorpus
    fi
fi


if [ "$FORMAT" == true ]; then
    # Extract articles from wikicorpus xml file into files of 100M each
    # This allows us to parallelize next step
    mkdir -p $DOWNLOAD_PATH/wikicorpus/data
    wikiextractor $DOWNLOAD_PATH/wikicorpus/wikicorpus_en.xml --bytes 25M \
        --processes $N_PROCESSES --output $DOWNLOAD_PATH/wikicorpus/data

    # Extract the data into text files where each line is a sentence and each
    # article is separated by a blank line. To prevent massive files,
    # articles/books are distributed across shards.
    python bert/format.py --dataset wikicorpus --processes $N_PROCESSES \
        --input_dir $DOWNLOAD_PATH/wikicorpus/data \
        --output_dir $FORMAT_PATH/wikicorpus --shards 256
    if [ "$INCLUDE_BOOKS" == true ]; then
        python bert/format.py --dataset bookscorpus --processes $N_PROCESSES \
            --input_dir $DOWNLOAD_PATH/bookscorpus/data \
            --output_dir $FORMAT_PATH/bookscorpus --shards 256
    fi
fi


if [ "$ENCODE" == true ]; then
    # RoBERTa Encoding:
    #   - no next sequence prediction
    #   - only use 512 length sequences
    python bert/encode_pretraining_data.py \
        --input_dir $FORMAT_PATH --output_dir $ENCODED_PATH \
        --vocab $VOCAB_FILE --max_seq_len 512 --short_seq_prob 0.1 \
        --next_seq_prob 0 --processes $N_PROCESSES
    # BERT Encoding:
    #   - next sequence prediction
    #   - two training phases (128 and 512 length sequences)
    #python bert/encode_pretraining_data.py \
    #    --input_dir $FORMAT_PATH --output_dir $ENCODED_PATH \
    #    --vocab $VOCAB_FILE --max_seq_len 128 --short_seq_prob 0.1 \
    #    --next_seq_prob 0.5 --processes $N_PROCESSES
    #python bert/encode_pretraining_data.py \
    #    --input_dir $FORMAT_PATH --output_dir $ENCODED_PATH \
    #    --vocab $VOCAB_FILE --max_seq_len 512 --short_seq_prob 0 \
    #    --next_seq_prob 0.5 --processes $N_PROCESSES
fi

