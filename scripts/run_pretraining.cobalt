#!/bin/bash
#COBALT -t 720
#COBALT -n 2
#COBALT -q full-node
#COBALT -A SuperBERT
#COBALT -O bert_pretrain_$jobid
##COBALT -M {YOUR EMAIL}
#COBALT --jobname bert-pretrain

# USAGE:
#   
#   To launch pretraining with this script, first provide the training
#   config, training data path, and output directory via the CONFIG,
#   DATA, and OUTPUT_DIR variables at the top of the script.
#
#   Run locally on a compute node:
#
#     $ ./run_pretraining.cobalt 
#
#   Submit as a Cobalt job (edit Cobalt arguments in the #COBALT directives
#   at the top of the script):
#
#     $ qsub-gpu path/to/run_pretraining.cobalt
#
#   Notes: 
#     - training configuration (e.g., # nodes, # gpus / node, etc.) will be
#       automatically inferred
#     - additional arguments to run_pretraining.py can be specified by
#       including them after run_pretraining.cobalt. E.g.,
#
#       $ ./run_pretraining.cobalt --steps 1000 --learning_rate 5e-4
#

PHASE=1

if [[ "$PHASE" -eq 1 ]]; then
    CONFIG=config/bert_pretraining_phase1_config.json
    DATA=/lus/theta-fs0/projects/SuperBERT/jgpaul/datasets/encoded/wikibooks/static_masked_30K/sequences_lowercase_max_seq_len_128_next_seq_task_true/
else
    CONFIG=config/bert_pretraining_phase1_config.json
    DATA=/lus/theta-fs0/projects/SuperBERT/jgpaul/datasets/encoded/wikibooks/static_masked_30K/sequences_lowercase_max_seq_len_128_next_seq_task_true/
fi

OUTPUT_DIR=results/bert_large_uncased_wikibooks_pretraining

# Figure out training environment
if [[ -z "${COBALT_NODEFILE}" ]]; then
    RANKS=$HOSTNAME
    NNODES=1
else
    MASTER_RANK=$(head -n 1 $COBALT_NODEFILE)
    RANKS=$(tr '\n' ' ' < $COBALT_NODEFILE)
    NNODES=$(< $COBALT_NODEFILE wc -l)
fi

# Commands to run prior to the Python script for setting up the environment
PRELOAD="source /etc/profile ; "
PRELOAD+="module load conda/pytorch ; "
PRELOAD+="conda activate /lus/theta-fs0/projects/SuperBERT/jgpaul/envs/pytorch-1.9.1-cu11.3 ; "
PRELOAD+="export OMP_NUM_THREADS=8 ; "

# torchrun launch configuration
LAUNCHER="python -m torch.distributed.run "
LAUNCHER+="--nnodes=$NNODES --nproc_per_node=auto --max_restarts 0 "
if [[ "$NNODES" -eq 1 ]]; then
    LAUNCHER+="--standalone "
else
    LAUNCHER+="--rdzv_backend=c10d --rdzv_endpoint=$MASTER_RANK "
fi

# Training script and parameters
CMD="run_pretraining.py --input_dir $DATA --output_dir $OUTPUT_DIR --config_file $CONFIG "

FULL_CMD=" $PRELOAD $LAUNCHER $CMD $@ "
echo "Training Command: $FULL_CMD"

# Launch the pytorch processes on each worker (use ssh for remote nodes)
RANK=0
for NODE in $RANKS; do
    if [[ "$NODE" == "$HOSTNAME" ]]; then
        echo "Launching rank $RANK on local node $NODE"
        eval $FULL_CMD &
    else
        echo "Launching rank $RANK on remote node $NODE"
        ssh $NODE "cd $PWD; $FULL_CMD" &
    fi
    RANK=$((RANK+1))
done

wait

