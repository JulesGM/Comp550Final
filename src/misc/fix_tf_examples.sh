#!/bin/bash
#SBATCH --job-name=fix_tf_examples
#SBATCH --ntasks=2
#SBATCH --time=30:00
#SBATCH --mem=32Gb
#SBATCH --output=/network/tmp1/chenant/sharing/comp-550/tmp/fixed_tf_logs/dec-10-a_nofilter_fixed_output-error.txt
#SBATCH --error=/network/tmp1/chenant/sharing/comp-550/tmp/fixed_tf_logs/dec-10-a_nofilter_fixed_output-error.txt
# ============================================================================
# Fix the pretraining tf-example files with Data Loss Error
#
# Run from inside of src folder
#
# Pre-run, set the appropriate parameters:
#   SBATCH --output : where to print the output items
#   SBATCH --error: where to print the error items
#   DATA_DIR: which directory (of potentially corrupted tf-example files) to
#             read from
#   OUT_DIR: which directory to write fixed tf-example files to
#   ..training related paramters (how many file out, batch size, etc.)
#
#
# Submut this file via sbatch
# ============================================================================
set -e # Close immidiately if a line returns an error.
set -u # Close immidiately if we try to access a variable that doesn't exist.


# ==
# Paths set-up

# File to take as input
DATA_DIR_BASE="/network/home/gagnonju/shared/data/parallel_jobs_logs"
DATA_DIR="$DATA_DIR_BASE/2019-12-09_filtered-out_nofilter"

# Directory to write to for output
OUT_DIR_BASE="/network/tmp1/chenant/sharing/comp-550/tmp/fixed_tf_logs"
OUT_DIR="$OUT_DIR_BASE/dec-10-a_nofilter_fixed"

# Number of output files to write
NUM_OUT_FILES="30"
# Batch size for reading input
BATCH_SIZE="256"

# Virtual environment
VENV_PATH="$SLURM_TMPDIR/cur_venv"

# BERT vocab
VOCAB_PATH="$SLURM_TMPDIR/vocab.txt"  # Path to the temp local BERT vocabulary file
VOCAB_URL="https://raw.githubusercontent.com/microsoft/BlingFire/master/ldbsrc/bert_base_cased_tok/vocab.txt"



# ==========
# Below should be automatic

# ==
# Write the output directory if not already present
if [ ! -d "$OUT_DIR" ] ; then
  mkdir -p "$OUT_DIR"
fi

# ==
# Get the BERT vocabulary file if not present
if [ ! -f "$VOCAB_PATH" ] ; then
  wget "$VOCAB_URL" -O "$VOCAB_PATH"
fi


# ==
# Do the same thing as the pretraining script does for BERT
module load cuda/10.0                                   # load gpu-related items
module load cuda-10.0/cudnn/7.3
module load python/3.7/tensorflow-gpu/1.15.0rc2

# ==
# Virtual environment set-up

if [ ! -d "$VENV_PATH" ] ; then
  virtualenv "$VENV_PATH"
fi
source "$VENV_PATH/bin/activate"

python -m pip install numpy
python -m pip install pandas    # this is not necessary
python -m pip install colorama
python -m pip install tqdm


# ==
# Run stuff

echo -e "\n=========="
echo "Start fix_tf_examples script"
echo -e "==========\n"

python -u ./misc/fix_tf_examples.py \
          --input_dir_path "$DATA_DIR" \
          --output_dir_path "$OUT_DIR" \
          --vocab_file_path "$VOCAB_PATH" \
          --num_out_files "$NUM_OUT_FILES" \
          --batch_size "$BATCH_SIZE"
