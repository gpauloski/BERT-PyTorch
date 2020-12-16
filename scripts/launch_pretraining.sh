#!/bin/bash

NGPUS=1
NNODES=1
MASTER=""
CONFIG=""

while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | sed 's/^[^=]*=//g'`
    if [[ "$VALUE" == "$PARAM" ]]; then
        shift
        VALUE=$1
    fi
    case $PARAM in
        -h|--help)
            echo "USAGE: ./launch_node_torch_imagenet.sh"
            echo "  -h,--help           Display this help message"
            echo "  -N,--ngpus [int]    Number of GPUs per node (default: 1)"
            echo "  -n,--nnodes [int]   Number of nodes this script is launched on (default: 1)"
            echo "  -m,--master [str]   Address of master node (default: \"\")"
            echo "  -c,--config [path]  Config file for training (default: \"\")"
            exit 0
        ;;
        -N|--ngpus)
            NGPUS=$VALUE
        ;;
        -n|--nnodes)
            NNODES=$VALUE
        ;;
        -m|--master)
            MASTER=$VALUE
        ;;
        -c|--config)
            CONFIG=$VALUE
        ;;
        *)
          echo "ERROR: unknown parameter \"$PARAM\""
          exit 1
        ;;
    esac
    shift
done

if [ -z "$OPMI_COMM_WORLD_RANK" ]; then
    LOCAL_RANK=$MV2_COMM_WORLD_RANK
else
    LOCAL_RANK=$OMPI_COMM_WORLD_RANK
fi

echo Launching torch.distributed: nproc_per_node=$NGPUS, nnodes=$NNODES, master_addr=$MASTER, local_rank=$LOCAL_RANK

python -m torch.distributed.launch \
   --nproc_per_node=$NGPUS --nnodes=$NNODES --node_rank=$LOCAL_RANK --master_addr=$MASTER \
   run_pretraining.py --config_file=$CONFIG

