# ===================================================================
# Extracts the bookcorpus on the Mila cluster
#
# Usage (after logging into an allocated node with $SLURM_TMPDIR
# as the temporary directory of that node):
#   bash extract_bookcorpus.sh
# ===================================================================
set -e # Close immidiately if a line returns an error.
set -u # Close immidiately if we try to access a variable that doesn't exist.

# Environmental variables
VENV_PATH="$SLURM_TMPDIR/cur_venv"              # path of vitual environment
BOOKCORPUS_PATH="$SLURM_TMPDIR/bookcorpus"      # bookcorpus code repository
DATA_DIR="/network/tmp1/chenant/class/comp-550" # directory for data storage
BOOKS_DIR="$DATA_DIR/bookcorpus-books"          # path to the output books


# Load module
module load python/3.7


# Create temporary virtual environment and activate
if [ ! -f "$VENV_PATH" ] ; then
  virtualenv $VENV_PATH
fi
source $VENV_PATH/bin/activate


# Clone the bookcorpus repository
if [ ! -f "$BOOKCORPUS_PATH" ] ; then
  mkdir $BOOKCORPUS_PATH
fi
git clone https://github.com/soskek/bookcorpus $BOOKCORPUS_PATH


# Install requirements for bookcorpus
python -m pip install -r $BOOKCORPUS_PATH/requirements.txt

# Get the list of book json (skipping because they already provide a list)
# python -u $BOOKCORPUS_PATH/download_list.py > \
#           $BOOKCORPUS_PATH/url_list.jsonl \

# Get the book
python $BOOKCORPUS_PATH/download_files.py \
       --list $BOOKCORPUS_PATH/url_list.jsonl \
       --out $BOOKCORPUS_PATH/out_txts \
       --trash-bad-count \


# Copy the books to an appropriate directory (uncomment below to copy)
# cp -r $BOOKCORPUS_PATH/out_txts $BOOKS_DIR



