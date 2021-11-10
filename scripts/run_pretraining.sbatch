#!/bin/bash
#SBATCH -J bert-pretrain           # Job name
#SBATCH -o bert-pretrain.o%j       # Name of stdout output file
#SBATCH -N 16                      # Total # of nodes 
#SBATCH -n 32                      # Total # of mpi tasks
#SBATCH -t 48:00:00                # Run time (hh:mm:ss)
#SBATCH --mail-user={YOUR EMAIL}
#SBATCH --mail-type=end            # Send email at begin and end of job
#SBATCH -p v100
#SBATCH -A Deep-Learning-at-Sca    # Allocation

#   Sample Slurm job script for TACC Longhorn Nodes
#   
#   To launch pretraining with this script, first provide the training
#   config, training data path, and output directory via the CONFIG,
#   DATA, and OUTPUT_DIR variables at the top of the script.
#
#   Run locally on a compute node:
#
#     $ ./run_pretraining.sbatch 
#
#   Submit as an Sbatch job (edit Sbatch arguments in the #SBATCH directives
#   at the top of the script):
#
#     $ qsub-gpu path/to/run_pretraining.sbatch
#
#   Notes: 
#     - training configuration (e.g., # nodes, # gpus / node, etc.) will be
#       automatically inferred
#     - additional arguments to run_pretraining.py can be specified by
#       including them after run_pretraining.cobalt. E.g.,
#
#       $ ./run_pretraining.sbatch --steps 1000 --learning_rate 5e-4
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
if [[ -z "${SLURM_NODELIST}" ]]; then
    RANKS=$HOSTNAME
    NNODES=1
else
    NODEFILE=/tmp/nodefile
    scontrol show hostnames $SLURM_NODELIST > $NODEFILE
    MASTER_RANK=$(head -n 1 $NODEFILE)
    RANKS=$(tr '\n' ' ' < $NODEFILE)
    NNODES=$(< $NODEFILE wc -l)
fi

# Commands to run prior to the Python script for setting up the environment
PRELOAD+="module load conda ; "
PRELOAD+="conda activate pytorch ; "
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

