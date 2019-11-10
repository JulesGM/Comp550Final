# ============================================================================
# Job script for cleaning the bookcorpus books
#
#
# ============================================================================
set -e # Close immidiately if a line returns an error.
set -u # Close immidiately if we try to access a variable that doesn't exist.

# Environmental variables
BOOKS_DIR="/network/tmp1/chenant/class/comp-550/tmp/out_txts"   # pre-downloaded directory of books
BOOKS="$SLURM_TMPDIR/books"                                     # local directory of books (to copy above to)
VENV_PATH="$SLURM_TMPDIR/cur_venv"                              # where to put the temporary virtual env
BOOKCORPUS_REPO="$SLURM_TMPDIR/bookcorpus-repo"                 # where to put the temporary bookcorpus repository
CLEAN_BOOKS="$SLURM_TMPDIR/cleaned-books"                       # processed books (cleaned plaintext)


# Load module
module load python/3.7

# Copy over pre-downloaded book text files
cp -r $BOOKS_DIR $BOOKS

# Set up and activate temporary virtualenv
if [ ! -d "$VENV_PATH" ] ; then
  virtualenv $VENV_PATH
fi
source $VENV_PATH/bin/activate

# Installing local requirements (the bookcorpus repository isn't complete)
python -m pip install numpy scipy pandas
python -m pip install nltk


# Get the bookcorpus repository and its requirements
if [ ! -d "$BOOKCORPUS_REPO" ] ; then
  mkdir $BOOKCORPUS_REPO
  git clone https://github.com/soskek/bookcorpus.git $BOOKCORPUS_REPO
fi
python -m pip install -r $BOOKCORPUS_REPO/requirements.txt


# testing
#python $BOOKCORPUS_REPO/make_sentlines.py $BOOKS > $SLURM_TMPDIR/tmp-allbooks.txt

# Testing actual script TODO rename and reformat
if [ ! -d "$CLEAN_BOOKS" ] ; then
  mkdir $CLEAN_BOOKS
fi
python bookcorpus-cleaning.py --input-dir $BOOKS \
                              --output-dir $CLEAN_BOOKS \
                              --min-sent-len 4 \
                              --remove-heads 0 \





