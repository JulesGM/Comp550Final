# ============================================================================
# Job script for generate tf examples
#
# Usage: bash src/clean_bookcorpus.sh
# ============================================================================
set -e # Close immidiately if a line returns an error.
set -u # Close immidiately if we try to access a variable that doesn't exist.

# Environmental variables
DATA_DIR="/network/tmp1/chenant/class/comp-550"   # general data directory
ID_BOOKS_DIR="$DATA_DIR/tmp/test_clean_id_books"  # downloaded books directory
LOC_BOOKS="$SLURM_TMPDIR/id-books"                # local books directory (to copy above to)
VENV_PATH="$SLURM_TMPDIR/cur_venv"                # temp virtual env directory
BOOKCORPUS_REPO="$SLURM_TMPDIR/bookcorpus-repo"   # temp Bookcorpus git repository

#LOC_CLEAN_BOOKS="$SLURM_TMPDIR/cleaned-id-books"  # local cleaned books dir
#CLEAN_BOOKS="$DATA_DIR/tmp/test_clean_id_books"   # cleaned books directory


# Load module
module load python/3.7

# Copy over pre-downloaded (raw) book text files
cp -r $ID_BOOKS_DIR "$LOC_BOOKS"

# Set up and activate temporary virtualenv
if [ ! -d "$VENV_PATH" ] ; then
  virtualenv $VENV_PATH
fi
source $VENV_PATH/bin/activate

# Installing local requirements (the bookcorpus repository isn't complete)
python -m pip install numpy scipy pandas
python -m pip install nltk
python -m pip install tensorflow

# Get the bookcorpus repository and its requirements
if [ ! -d "$BOOKCORPUS_REPO" ] ; then
  mkdir $BOOKCORPUS_REPO
  git clone https://github.com/soskek/bookcorpus.git $BOOKCORPUS_REPO
fi
python -m pip install -r "$BOOKCORPUS_REPO/requirements.txt"


# Generate the tf examples
#if [ ! -d "$LOC_CLEAN_BOOKS" ] ; then
#  mkdir $LOC_CLEAN_BOOKS
#fi
python src/build-ex-from-id.py --input-dir "$LOC_BOOKS" \
                               --output-dir None \
                               --shuf-sentences False \
                               --sent-per-book -1 \





# Copy local temporary cleaned books to data directory (uncomment to copy)
# cp -r $LOC_CLEAN_BOOKS $CLEAN_BOOKS
