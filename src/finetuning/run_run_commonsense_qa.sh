set -e
set -u

VENV_PATH="$SLURM_TMPDIR/cur_venv"
COMMONSENSEQA="$SLURM_TMPDIR/commonsenseqa"
BERT_URL="https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip"
BERT_FILE_NAME=$(python3 -c "print(\"$BERT_URL\".split(\"/\")[-1].split(\".\")[0])")
# VOCAB_URL="https://raw.githubusercontent.com/microsoft/BlingFire/master/ldbsrc/bert_base_cased_tok/vocab.txt"

echo $BERT_FILE_NAME
BERT_DIR="$SLURM_TMPDIR/bert_model"

if [ ! -d "$COMMONSENSEQA" ] ; then
  mkdir "$COMMONSENSEQA"
fi

# Download the dataset
cp train_rand_split.jsonl "$COMMONSENSEQA"/train_rand_split.jsonl
cp dev_rand_split.jsonl "$COMMONSENSEQA"/dev_rand_split.jsonl
cp test_rand_split.jsonl "$COMMONSENSEQA"/test_rand_split.jsonl


echo -e "\n###############################################################"
echo "# Load tensorflow gpu"
echo "###############################################################"
# Load python
module purge
module refresh
module load cuda/10.0
module load cuda/10.0/cudnn/7.6
module load python/3.7/tensorflow-gpu/1.15.0rc2

echo -e "\n###############################################################"
echo "# Activate VENV"
echo "###############################################################"
#rm -rf "$VENV_PATH" || true
# Set up and activate temporary virtualenv
rm -rf "$VENV_PATH" || true
if [ ! -d "$VENV_PATH" ] ; then
  virtualenv "$VENV_PATH"
fi
source "$VENV_PATH/bin/activate"
python -m pip install tqdm

echo -e "\n###############################################################"
echo "# Maybe Download the BERT model"
echo "###############################################################"
if [ ! -d "$BERT_DIR" ] ; then
  mkdir "$BERT_DIR"
fi

if [ ! -f "$BERT_DIR/$BERT_FILE_NAME.zip" ] ; then
  wget "$BERT_URL" -O "$BERT_DIR/$BERT_FILE_NAME.zip"
  unzip "$BERT_DIR/$BERT_FILE_NAME.zip" -d "$BERT_DIR/"
  mv "$BERT_DIR/$BERT_FILE_NAME"/* "$BERT_DIR"
fi

rm -rf "$COMMONSENSEQA/checkpoints"
if [ ! -d "$COMMONSENSEQA/checkpoints" ] ; then
  mkdir "$COMMONSENSEQA/checkpoints"
fi

echo -e "\n###############################################################"
echo "# Running run_commonsense_qa.py"
echo "###############################################################"

python run_commonsense_qa.py \
  --data_dir="$COMMONSENSEQA"\
  --split=rand \
  --bert_config_file="$BERT_DIR/bert_config.json" \
  --vocab_file="$BERT_DIR/vocab.txt" \
  --output_dir="$COMMONSENSEQA/checkpoints" \
  --init_checkpoint="$BERT_DIR/bert_model.ckpt" \
  --do_lower_case=False \
  --max_seq_length=128 \
  --do_train=True \
  --do_eval=True \
  --do_predict=False \
  --train_batch_size=8 \
  --eval_batch_size=8 \
  --predict_batch_size=8 \
