# ============================================================================
# Job script for cleaning the bookcorpus books
#
# Usage: bash src/clean_bookcorpus.sh
# ============================================================================
set -e # Close immidiately if a line returns an error.
set -u # Close immidiately if we try to access a variable that doesn't exist.

BERT_LIB_PATH="bert"
# Environmental variables
DATA_DIR="/network/home/gagnonju/shared/data"   # general data directory
BOOKS_DIR="/network/tmp1/chenant/sharing/comp-550/bookcorpus/raw_books/out_txts"          # downloaded books directory
BOOKCORPUS_REPO="$SLURM_TMPDIR/bookcorpus-repo"   # temp Bookcorpus git repository
LOC_CLEAN_BOOKS="$DATA_DIR/cleaned-id-books"  # local cleaned books dir
CLEAN_BOOKS="$DATA_DIR/tmp/test_clean_id_books"   # cleaned books directory
#LOC_BOOKS="$DATA_DIR/loc_books"
VOCAB_PATH="$SLURM_TMPDIR/vocab.txt"  # Path to the temp local BERT vocabulary file
VOCAB_URL="https://raw.githubusercontent.com/microsoft/BlingFire/master/ldbsrc/bert_base_cased_tok/vocab.txt"

# source venv_setup.sh

# Get vocabulary file
if [ ! -f "$VOCAB_PATH" ] ; then
  echo -e "\n###########################################################"
  echo "# Downloading the BERT vocab"
  echo "###########################################################"
  wget "$VOCAB_URL" -O "$VOCAB_PATH"
fi

echo -e "\n###########################################################"
echo "# Fixing BERT for TF 2.0"
echo "###########################################################"
# Minor compatibility adjustment for the BERT code.
find "$BERT_LIB_PATH" -iname "*.py" -exec sed -i 's/tf.gfile./tf.io.gfile./g' "{}"  \;

echo -e "\n###########################################################"
echo "# Running bookcorpus-cleaning.py"
echo "###########################################################"
# Clean up the raw book texts
if [ ! -d "$LOC_CLEAN_BOOKS" ] ; then
  mkdir "$LOC_CLEAN_BOOKS"
fi
python bookcorpus_cleaning.py --input-dir "$BOOKS_DIR" \
                              --output-dir "$LOC_CLEAN_BOOKS" \
                              --min-sent-len 4 \
                              --remove-blank True \
                              --remove-heads 0 \
                              --id-seq-length 128 \
                              --oov-id 100 \
                              --vocab_path "$VOCAB_PATH" \
                              --base-tok-file "bert_base_cased_tok.bin" \
                              --mode "blingfire" \
                              # --max_number_of_books 50 # WARNS
