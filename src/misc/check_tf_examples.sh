#!/bin/bash
#SBATCH --job-name=check_tf_examples
#SBATCH --ntasks=2
#SBATCH --time=30:00
#SBATCH --mem=32Gb
#SBATCH --output=/network/tmp1/chenant/sharing/comp-550/check_tf_logs/temp_output.txt
#SBATCH --error=/network/tmp1/chenant/sharing/comp-550/check_tf_logs/temp_error.txt
# ============================================================================
# Check that the pre-training tf-example files are in good shape
#
# Run from inside of src folder
#
# Pre-run, set the appropriate parameters:
#   SBATCH --output : where to write output .txt file
#   SBATCH --error : where to write error .txt file
#   DATA_DIR: path to the input directory (of tf-examples)
#
# Submut this file via sbatch
# ============================================================================
set -e # Close immidiately if a line returns an error.
set -u # Close immidiately if we try to access a variable that doesn't exist.


# ==
# Paths set-up

# File to take as input
DATA_DIR_BASE="/network/home/gagnonju/shared/data/parallel_jobs_logs"
DATA_DIR="$DATA_DIR_BASE/temp_dir"

# TODO (potential): save dataframe

# Virtual environment
VENV_PATH="$SLURM_TMPDIR/cur_venv"


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
python -m pip install pandas
python -m pip install colorama
python -m pip install tqdm


# ==
# Run stuff

echo -e "\n=========="
echo "Start check_tf_examples script"
echo -e "==========\n"

python -u ./misc/check_tf_examples.py \
          --input_dir_path "$DATA_DIR" \
