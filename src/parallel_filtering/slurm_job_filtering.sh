#!/bin/bash
# ==
# Assumes it'll be run from the src/ directory.
#

# TODO set -e and stuff


PY_SCRIPT="../filter_inference.py" # TODO change this


DATA_DIR="/network/tmp1/chenant/sharing/comp-550"   # general data directory
FILTERED_OUTPUT_PATH="$DATA_DIR/bookcorpus/ TODO WRITE THIS"


MODEL_SAVE_PATH_ORI="TODO This"

OUTPUT_DIR_PATH=$out_dir




NUM_OUT_FILE_PER_SHARD="10"
NUM_SHARDS=$n_shards
START_SHARD_IDX=$start_shard_idx
SHARD_QUANTITY=$shard_quant





# Local and relative paths
MODEL_SAVE_PATH_LOC="$SLURM_TMPDIR/model_nbc.pkl"
MODEL_CONFIG_PATH_INFERENCE="../configs/nbc_inference.json"
VOCAB_URL="https://raw.githubusercontent.com/microsoft/BlingFire/master/ldbsrc/bert_base_cased_tok/vocab.txt"
VOCAB_PATH="$SLURM_TMPDIR/vocab.txt"

#
FORCE="True" # "True" or "False"


# ==
# Download an example BERT vocab if it doesn't exist.
if [[ "$FORCE" == "True" ]] || [ ! -f "$VOCAB_PATH" ] ; then
    echo -e "\n####################################################"
    echo "Downloading the BERT vocab"
    echo "####################################################"
    wget "$VOCAB_URL" -O "$VOCAB_PATH"
fi



for ((i=0; i<NUM_SHARDS; i++)); do
  # Compute the sharding index
  CUR_SHARD_IDX=$(($i + $START_SHARD_IDX))

  python $PY_SCRIPT --filter_type=no \
        --batch_size=10 -v=0 \
        --num_map_threads=1 \
        --shuffle_buffer_size=1 \
        --output_data_path="$FILTERED_OUTPUT_PATH" \
        --input_data_path="$UNLABELED_DIR" \
        --json_config_path="$MODEL_CONFIG_PATH_INFERENCE" \
        --vocab_path="$VOCAB_PATH" \
        --model_ckpt_path="$MODEL_SAVE_PATH_LOC" \
        --num_output_shards=$NUM_OUT_FILE_PER_SHARD \
        --sharding_quantity=$SHARD_QUANTITY \
        --sharding_idx $CUR_SHARD_IDX \
        &



  python -u $PY_SCRIPT \
            --sharding_quantity $SHARD_QUANTITY \
            --sharding_idx $CUR_SHARD_IDX \
            &
done




#SBATCH --partition=long
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/network/tmp1/chenant/sharing/comp-550/bert-pretrain/dec-8_out/output-pretrain.txt
#SBATCH --error=/network/tmp1/chenant/sharing/comp-550/bert-pretrain/dec-8_out/error-pretrain.txt
