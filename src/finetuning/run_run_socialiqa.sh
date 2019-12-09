set -e
set -u

VENV_PATH="$SLURM_TMPDIR/cur_venv"
SOCIALIQA="$SLURM_TMPDIR/socialiqa"
BERT_URL="https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip"
BERT_FILE_NAME=$(python3 -c "print(\"$BERT_URL\".split(\"/\")[-1].split(\".\")[0])")
echo $BERT_FILE_NAME
BERT_DIR="$SLURM_TMPDIR/bert_model"

if [ ! -d "$SOCIALIQA" ] ; then
  mkdir "$SOCIALIQA"
fi

# Download the dataset
wget https://maartensap.github.io/social-iqa/data/socialIQa_v1.4.tgz -O "$SOCIALIQA"/socialIQa_v1.4.tgz

# Unzip the dataset
tar -xf "$SOCIALIQA/socialIQa_v1.4.tgz" -C "$SOCIALIQA"

echo -e "\n###############################################################"
echo "# Load tensorflow gpu"
echo "###############################################################"
# Load python
module purge
module refresh
module load python/3.7
module load cuda/10.0
module load cuda/10.0/cudnn/7.6
module load python/3.7/tensorflow-gpu/1.15.0rc2

echo -e "\n###############################################################"
echo "# Activate VENV"
echo "###############################################################"
rm -rf "$VENV_PATH" || true
# Set up and activate temporary virtualenv
if [ ! -d "$VENV_PATH" ] ; then
  virtualenv "$VENV_PATH"
fi
source "$VENV_PATH/bin/activate"
python -m pip install tqdm

echo -e "\n###############################################################"
echo "# Load CUDA and CUDNN "
echo "###############################################################"


echo -e "\n###############################################################"
echo "# convert_socialiqa.py"
echo "###############################################################"
# Convert the dataset to the format expected by the training script
python convert_socialiqa.py "$SOCIALIQA" "$SOCIALIQA"

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


rm -rf "$SOCIALIQA/checkpoints"
if [ ! -d "$SOCIALIQA/checkpoints" ] ; then
  mkdir "$SOCIALIQA/checkpoints"
fi

echo -e "\n###############################################################"
echo "# Running run_socialiqa.py"
echo "###############################################################"

python run_socialiqa.py \
  --data_dir="$SOCIALIQA"\
  --bert_config_file="$BERT_DIR/bert_config.json" \
  --vocab_file="$BERT_DIR/vocab.txt" \
  --output_dir="$SOCIALIQA/checkpoints" \
  --init_checkpoint="$BERT_DIR/bert_model.ckpt" \
  --do_lower_case=False \
  --max_seq_length=128 \
  --do_train=True \
  --do_eval=True \
  --do_predict=False \
  --train_batch_size=8 \
  --eval_batch_size=8 \
  --predict_batch_size=8 \
