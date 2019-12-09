#!/bin/bash
#SBATCH --partition=low
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/network/tmp1/chenant/sharing/comp-550/bookcorpus/dec-7_extract/output_dec-7_extract.txt
#SBATCH --error=/network/tmp1/chenant/sharing/comp-550/bookcorpus/dec-7_extract/error_dec-7_extract.txt

# ===================================================================
# Extracts the bookcorpus on the Mila cluster
#
# Usage (after logging into an allocated node with $SLURM_TMPDIR
# as the temporary directory of that node):
#   bash extract_bookcorpus.sh
#
# Or via sbatch: sbatch extract_bookcorpus.sh
# ===================================================================
set -e # Close immidiately if a line returns an error.
set -u # Close immidiately if we try to access a variable that doesn't exist.

# Environmental variables
VENV_PATH="$SLURM_TMPDIR/cur_venv"              # path of vitual environment
BOOKCORPUS_PATH="$SLURM_TMPDIR/bookcorpus"      # bookcorpus code repository
DATA_DIR="/network/tmp1/chenant/sharing/comp-550/bookcorpus" # directory for data storage
BOOKS_DIR="$DATA_DIR/dec-7_extract/out_txts"      # path to the output books



# Get the list of book json (skipping because they already provide a list)
# python -u $BOOKCORPUS_PATH/download_list.py > \
#           $BOOKCORPUS_PATH/url_list.jsonl \

module load python/3.7

# Set up and activate temporary virtualenv
if [ ! -d "$VENV_PATH" ] ; then
  virtualenv "$VENV_PATH"
fi
source "$VENV_PATH/bin/activate"

# Installing local requirements (the bookcorpus repository isn't complete)
python -m pip install numpy scipy pandas tqdm pygments colored_traceback blingfire nltk colorama -q

# Get the bookcorpus repository and its requirements
if [ ! -d "$BOOKCORPUS_PATH" ] ; then
  mkdir "$BOOKCORPUS_PATH"
  git clone https://github.com/soskek/bookcorpus.git "$BOOKCORPUS_PATH"
fi
python -m pip install -r "$BOOKCORPUS_PATH/requirements.txt" -q

# Hackerman (changing sleep time)
sed -i 's/SUCCESS_SLEEP_SEC = 0\.001/SUCCESS_SLEEP_SEC = 1/g' \
    $BOOKCORPUS_PATH/download_files.py


# Get the books in plaintext form
python -u $BOOKCORPUS_PATH/download_files.py \
          --list $BOOKCORPUS_PATH/url_list.jsonl \
          --out $BOOKS_DIR \
          --trash-bad-count \

# Copy the books to an appropriate directory (uncomment below to copy)
# cp -r $BOOKCORPUS_PATH/out_txts $BOOKS_DIR



