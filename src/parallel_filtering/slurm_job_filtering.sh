#!/bin/bash
# ============================================================================
# Slurm sript to be submitted to run the filter inference that produces
# additional fitered tf-example files for BERT pre-training
#
# Assumes it will be ran from Comp550Final/src
#
# ============================================================================
set -e # Close immidiately if a line returns something else than 0 (aka, if there is an error)
set -u # Close immidiately if we try to access a variable that doesn't exist.


# ==
# Paths

# Python inference filtering script to be parallized
PY_SCRIPT="./filter_inference.py"

# Path to the input unlabelled directory (to be filtered, read from)
UNLABELED_DIR="$in_dir"           # passed from job submission script

# Path to the output directory (filtered tf-examples, directory to write to)
FILTERED_OUTPUT_PATH="$out_dir"   # passed from job submissions script

# Type of model to run
MODEL_TYPE="$mod_type"            # passed from job submission script
# Where the trained model .pkl file is saved (read from)
MODEL_SAVE_PATH_LOC="$mod_pkl"    # passed from job submission script
# Where the model configuration is (read from)
MODEL_CONFIG_PATH_INFERENCE="$mod_config" # passed form job submission script

# Paths to download BERT's vocabulary files
VOCAB_PATH="/network/home/gagnonju/shared/data/vocab.txt"


# ==
# Options
FORCE="True" # "True" or "False"

NUM_OUT_FILE_PER_SHARD="10"
NUM_SHARDS="$n_shards"              # passed from job submission script
START_SHARD_IDX="$start_shard_idx"  # passed from job submission script
SHARD_QUANTITY="$shard_quant"       # passed from job submission script

# ==
# Set-up the environment
source ./parallel_filtering/conda_venv_setup.sh



echo -e "\n####################################################"
echo "Running filtering"
echo "####################################################"
#export CUDA_VISIBLE_DEVICES=""

for ((i=0; i<NUM_SHARDS; i++)); do
  # Compute the sharding index
  CUR_SHARD_IDX=$(("$i" + "$START_SHARD_IDX"))
  echo "Launching $i $PY_SCRIPT" ...
  python $PY_SCRIPT \
        --filter_type="$MODEL_TYPE" \
        --batch_size=1024 \
        -v=0 \
        --num_map_threads=1 \
        --shuffle_buffer_size=1 \
        --input_data_glob_pattern="$UNLABELED_DIR" \
        --output_data_path="$FILTERED_OUTPUT_PATH" \
        --json_config_path="$MODEL_CONFIG_PATH_INFERENCE" \
        --vocab_path="$VOCAB_PATH" \
        --model_ckpt_path="$MODEL_SAVE_PATH_LOC" \
        --num_output_shards="$NUM_OUT_FILE_PER_SHARD" \
        --sharding_quantity="$SHARD_QUANTITY" \
        --sharding_idx="$CUR_SHARD_IDX" \
        &
    # save the pid
    pids[${i}]=$!
done

# Wait until all the jobs are done
for pid in ${pids[*]}; do
    echo "waiting on $pid"
    wait $pid
    echo "done waiting on $pid"
done
echo "DONE"