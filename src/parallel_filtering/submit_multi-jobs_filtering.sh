#!/bin/bash
# ============================================================================
# Submit multiple jobs for smart filtering to generate tf-example files
# for additional BERT pre-training
#
# Assume we are running this from inside Comp550Final/src
#
# The main variables to change are:
#   IN_DIR_PATH: path to the directory containing unfiltered, unmasked
#                tf-example files to be filtered
#   OUT_DIR_PATH: path to the output directory to deposit the filtered,
#                 masked, tf-example files (TODO confirm that output is masked)
#   MOD_TYPE: type of model to run (e.g. no_filter, nbc, etc.)
#   MOD_PKL: path to the trained model .pkl file (to do the filtering)
#   MOD_CONFIG: path to the model configuration file
# ============================================================================
set -e # Close immidiately if a line returns something else than 0 (aka, if there is an error)
set -u # Close immidiately if we try to access a variable that doesn't exist.


# ==
# Paths

# Input directory containing unfiltered, unmasked tf examples CONFIRM THIS
IN_DIR_GLOB='/network/home/gagnonju/shared/data/tf_examples_dir/*'

MOD_TYPE="lstm" #nbc, lstm, no

# Output directory to deposit the filtered tf examples
out_dir_name="`date +"%Y-%m-%d"`_filtered-out_$MOD_TYPE"

if [[ "$USER" == "chenant" ]] ; then
  echo "CHENANT MODE"
  OUT_DIR_PATH="/network/tmp1/chenant/sharing/comp-550/filter_models/inference/$out_dir_name"
elif [[ "$USER" == "gagnonju" ]] ; then
  echo "GAGNONJU MODE"
  OUT_DIR_PATH="/network/home/gagnonju/shared/data/parallel_jobs_logs/$out_dir_name"
else
  echo "GOT UNKOWN USER: $USER"
  exit
fi

# Path to the trained model pkl and confirguration

MOD_PKL="/network/home/gagnonju/shared/models/model_${MOD_TYPE}.pkl"
MOD_CONFIG="../configs/${MOD_TYPE}_inference.json"

# Job file to be submitted in parallel
SLURM_FILE_PATH="./parallel_filtering/slurm_job_filtering.sh"

# ==
# Options

NUM_JOBS=5        # number of sbatch jobs to submit
SHARD_PER_JOB=4   # number of shards per job (usually 4 for 4-core node)

PARTITION="long"  # long (low priority0 so we can submit multiple jobs
MEM_PER_JOB="16G"
GRES_PER_JOB="gpu:pascal:1"

TIME_PER_JOB="1:00:00" # time allowed per job




# ==
# No need to modify below - counter variables
SHARDING_QUANTITY=$(($NUM_JOBS * $SHARD_PER_JOB))
SHARDING_IDX=0


# ==
# Create output directory
#
if [ ! -d "$OUT_DIR_PATH" ] ; then
  echo "Creating output directory at: $OUT_DIR_PATH"
  mkdir -p $OUT_DIR_PATH
  echo "Directory created at $(date)" >> "$OUT_DIR_PATH/creation.txt"
fi



# ==
# Submit parallel jobs
#
#for ((i=1;i<=NUM_JOBS;i++)); do
  i=4
  SHARDING_IDX=$(expr "$i" \* "$SHARD_PER_JOB")
  # ==
  # Create the error and output file for each job
  shard_name="shard-${SHARDING_IDX}_of_${SHARDING_QUANTITY}"
  cur_error_file="$OUT_DIR_PATH/error_${shard_name}.txt"
  cur_out_file="$OUT_DIR_PATH/output_${shard_name}.txt"

  job_name="${shard_name}_parallel-job"

  # ==
  # Submit job
  sbatch --cpus-per-task=$SHARD_PER_JOB \
         --partition=$PARTITION \
         --mem=$MEM_PER_JOB \
         --gres=$GRES_PER_JOB \
         --time=$TIME_PER_JOB \
         --output="$cur_out_file" \
         --error="$cur_error_file" \
         --export=in_dir="$IN_DIR_GLOB",mod_type="$MOD_TYPE",mod_pkl=$MOD_PKL,mod_config="$MOD_CONFIG",out_dir="$OUT_DIR_PATH",shard_quant=$SHARDING_QUANTITY,n_shards="$SHARD_PER_JOB",start_shard_idx="$SHARDING_IDX" \
         --job-name="$job_name" \
         "$SLURM_FILE_PATH"



  # Increment the sharding index

#done
