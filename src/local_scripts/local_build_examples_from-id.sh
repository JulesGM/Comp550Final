# ============================================================================
# Job script to generate tf examples from .npy book files
#
# Specifically, it reads a directory of .npy book files (each file representing
# one book from the bookcorpus, where each row is a sentence and the content
# of the row are the token ids of that sentence).
# Then it builds tf examples of training examples for BERT training. Training
# example contain two sentences (in their token id form) concatenated together,
# with a 0.5 prob that the second sentence is sampled randomly from any book
# in the input corpus directory.
#
# Output a number of tf example file.
#
#
# Usage: bash src/clean_bookcorpus.sh
# ============================================================================
set -e # Close immidiately if a line returns an error.
set -u # Close immidiately if we try to access a variable that doesn't exist.

# Variables
DATA_DIR="/Users/jules/Documents/COMP550/data"   # general data directory
ID_BOOKS_DIR="$DATA_DIR/cleaned-id-books"  # downloaded books directory
TF_OUT_DIR="$DATA_DIR/tf_examples_dir" # output directory storing the tf example files

TMP_DATA_DIR="/tmp"
VENV_PATH="$TMP_DATA_DIR/cur_venv"       # temp virtual env directory
BOOKCORPUS_REPO="$TMP_DATA_DIR/bookcorpus-repo"   # temp Bookcorpus git repository

VOCAB_PATH="$TMP_DATA_DIR/vocab.txt"  # Path to the temp local BERT vocabulary file
VOCAB_URL="https://raw.githubusercontent.com/microsoft/BlingFire/master/ldbsrc/bert_base_cased_tok/vocab.txt"

# Get the bookcorpus repository and its requirements
if [ ! -d "$BOOKCORPUS_REPO" ] ; then
  mkdir $BOOKCORPUS_REPO
  git clone https://github.com/soskek/bookcorpus.git $BOOKCORPUS_REPO
fi
python -m pip install -r "$BOOKCORPUS_REPO/requirements.txt"

# Get vocabulary file
if [ ! -f "$VOCAB_PATH" ] ; then
  wget "$VOCAB_URL" -O "$VOCAB_PATH"
fi

# Create (temporary) directory to store the tf examples
if [ ! -d "$TF_OUT_DIR" ] ; then
  mkdir "$TF_OUT_DIR"
fi

# Generate the tf examples
python -u build_ex_from_id.py --input-dir "$ID_BOOKS_DIR" \
                               --output-dir "$TF_OUT_DIR" \
                               --vocab-file "$VOCAB_PATH" \
                               --shuf-sentences False \
                               --sent-per-book -1 
