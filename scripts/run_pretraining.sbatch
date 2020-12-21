#!/bin/bash

# Sample Slurm job script
#   for TACC Longhorn Nodes
#
#------------------------------------------------------------------------------

#SBATCH -J bertkfc                 # Job name
#SBATCH -o sbatch_logs/bertkfc.o%j # Name of stdout output file
#SBATCH -N 16                      # Total # of nodes 
#SBATCH -n 32                      # Total # of mpi tasks
#SBATCH -t 48:00:00                # Run time (hh:mm:ss)
#SBATCH --mail-user=jgpauloski@utexas.edu
#SBATCH --mail-type=end            # Send email at begin and end of job
#SBATCH -p v100
#SBATCH -A Deep-Learning-at-Sca    # Allocation

mkdir -p sbatch_logs

module load conda
conda activate pytorch
module unload spectrum_mpi
module use /home/01255/siliu/mvapich2-gdr/modulefiles/
module load gcc/7.3.0 
module load mvapich2-gdr/2.3.4

export MV2_USE_CUDA=1
export MV2_ENABLE_AFFINITY=1
export MV2_THREADS_PER_PROCESS=2
export MV2_SHOW_CPU_BINDING=1
export MV2_CPU_BINDING_POLICY=hybrid
export MV2_HYBRID_BINDING_POLICY=spread
export MV2_USE_RDMA_CM=0
export MV2_SUPPORT_DL=1

HOSTFILE=hostfile
cat $HOSTFILE

MASTER_RANK=$(head -n 1 $HOSTFILE)
NODES=$(< $HOSTFILE wc -l)
PROC_PER_NODE=4

# PHASE 1
mpirun -np $NODES -hostfile $HOSTFILE  bash scripts/launch_pretraining.sh  \
    --ngpus $PROC_PER_NODE --nnodes $NODES --master $MASTER_RANK \
    --config config/bert_pretraining_phase1_config.json \
	--input data/hdf5/lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en/ \
    --output results/bert_pretraining

# PHASE 2
mpirun -np $NODES -hostfile $HOSTFILE  bash scripts/launch_pretraining.sh  \
    --ngpus $PROC_PER_NODE --nnodes $NODES --master $MASTER_RANK \
    --config config/bert_pretraining_phase2_config.json \
	--input data/hdf5/lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en/ \
    --output results/bert_pretraining

