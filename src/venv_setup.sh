set -e # Close immidiately if a line returns something else than 0 (aka, if there is an error)
set -u # Close immidiately if we try to access a variable that doesn't exist.

BOOKCORPUS_REPO="$SLURM_TMPDIR/bookcorpus-repo"   # temp Bookcorpus git repository
VENV_PATH="$SLURM_TMPDIR/cur_venv"

echo "VENV_PATH: \"$VENV_PATH\""
echo "PYTHON EXECUTABLE: $(which python)"

echo -e "\n###########################################################"
echo "# Installing python"
echo "###########################################################"
module load python/3.7/tensorflow-gpu/2.0.0

echo -e "\n###########################################################"
echo "# Building and activating the VENV"
echo "###########################################################"
# Set up and activate temporary virtualenv
if [ ! -d "$VENV_PATH" ] ; then
  virtualenv "$VENV_PATH"
fi
source "$VENV_PATH/bin/activate"

# Installing local requirements (the bookcorpus repository isn't complete)
echo -e "\n###########################################################"
echo "# Installing the python third party dependencies"
echo "###########################################################"
python -m pip install numpy tqdm pygments colored_traceback \
          spacy blingfire spacy nltk colorama fire scikit-learn -q
python -m spacy download en_core_web_sm -q

# Get the bookcorpus repository and its requirements
if [ ! -d "$BOOKCORPUS_REPO" ] ; then
  mkdir "$BOOKCORPUS_REPO"
  git clone https://github.com/soskek/bookcorpus.git "$BOOKCORPUS_REPO"
fi
python -m pip install -r "$BOOKCORPUS_REPO/requirements.txt" -q

echo -e "\n###########################################################"
echo "# Installing cuda, cudnn and tensorflow-gpu"
echo "###########################################################"
module load cuda/10.0
module load cuda/10.0/cudnn/7.3


