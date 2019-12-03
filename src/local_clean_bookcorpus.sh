# ============================================================================
# Job script for cleaning the bookcorpus books
#
# Usage: bash src/clean_bookcorpus.sh
# ============================================================================
set -e # Close immidiately if a line returns an error.
set -u # Close immidiately if we try to access a variable that doesn't exist.

SCRIPTS_DIR="/Users/jules/Documents/COMP550/Comp550Final/src"

# Environmental variables
DATA_DIR="/Users/jules/Documents/COMP550/data"   # general data directory
BOOKS_DIR="$DATA_DIR/out_txts"          # downloaded books directory
LOC_CLEAN_BOOKS="$DATA_DIR/cleaned-id-books"  # local cleaned books dir

TMP_DATA_DIR="/tmp"
LOC_BOOKS="$TMP_DATA_DIR/books"                   # local books directory (to copy above to)
BOOKCORPUS_REPO="$TMP_DATA_DIR/bookcorpus-repo"   # temp Bookcorpus git repository

VOCAB_PATH="/tmp/vocab.txt"
VOCAB_URL="https://raw.githubusercontent.com/microsoft/BlingFire/master/ldbsrc/bert_base_cased_tok/vocab.txt"
if [ !  -f "$VOCAB_PATH" ] ; then
    echo -e "\n####################################################"
    echo "Downloading the BERT vocab"
    echo "####################################################"
    wget "$VOCAB_URL" -O "$VOCAB_PATH"
fi

if [ ! -d "$BOOKS_DIR" ] ; then
  >&2 echo "Error: The directory where the books should be doesn't exist."
fi

# Copy over pre-downloaded (raw) book text files
cp -r $BOOKS_DIR "$LOC_BOOKS"

echo "#################################################################"
echo -e "# cloning bookcorpus.git and installing requirements"
echo "#################################################################"
# Get the bookcorpus repository and its requirements
if [ ! -d "$BOOKCORPUS_REPO" ] ; then
  mkdir $BOOKCORPUS_REPO
  git clone https://github.com/soskek/bookcorpus.git $BOOKCORPUS_REPO
fi
python -m pip install -r "$BOOKCORPUS_REPO/requirements.txt"

echo "#################################################################"
echo -e "# Maybe making dirs"
echo "#################################################################"
if [ ! -d "$LOC_BOOKS" ] ; then
  mkdir $LOC_BOOKS
fi

if [ ! -d "$LOC_BOOKS" ] ; then
  mkdir $LOC_BOOKS
fi

# Clean up the raw book texts
if [ ! -d "$LOC_CLEAN_BOOKS" ] ; then
  mkdir $LOC_CLEAN_BOOKS
fi

echo "#################################################################"
echo -e "# python src/bookcorpus-cleaning.py"
echo "#################################################################"
python "$SCRIPTS_DIR"/bookcorpus-cleaning.py --input-dir "$LOC_BOOKS" \
                                  --output-dir "$LOC_CLEAN_BOOKS" \
                                  --min-sent-len 4 \
                                  --remove-blank True \
                                  --remove-heads 0 \
                                  --id-seq-length 128 \
                                  --oov-id 100 \
                                  --vocab_path "$VOCAB_PATH" \
                                  --mode "bert-native"