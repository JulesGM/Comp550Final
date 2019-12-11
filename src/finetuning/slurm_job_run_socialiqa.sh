#!/bin/bash
# =============================================================================
# Fine-tune a BERT model on the Social-IQA dataset.
#
# Assume we will be running this from Comp550Final/src/
#
# Also, this script has no ability to download a bert model (only use local)
#
# =============================================================================
set -e # Close immidiately if a line returns an error.
set -u # Close immidiately if we try to access a variable that doesn't exist.

# ==
# Variable set-up

# BERT model locations
#BERT_TRAINED_BASE="/network/tmp1/chenant/sharing/comp-550/bert-pretrain/2019-12-11_pretrain_bert_pt3_lr2e-3"
#BERT_TRAINED_DIR="$BERT_TRAINED_BASE/lstm-filter/temp"  # debug
BERT_TRAINED_DIR=$data_dir

# BERT model name
BERT_CHECKPT_NAME="model.ckpt"

# Output fine-tuned model directory location
#OUT_DIR_BASE="/network/tmp1/chenant/sharing/comp-550/fine-tune"
#OUT_DIR="$OUT_DIR_BASE/test-dec-10"    # debug
OUT_DIR=$out_dir



# ==
# Local variables (shouldn't have to touch)
VENV_PATH="$SLURM_TMPDIR/cur_venv"
SOCIALIQA="$SLURM_TMPDIR/socialiqa"
BERT_DIR_ORI="$SLURM_TMPDIR/orig_bert_model"
BERT_DIR_LOC="$SLURM_TMPDIR/our_bert_model"

# Url to get BERT model
BERT_URL="https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip"
BERT_FILE_NAME=$(python3 -c "print(\"$BERT_URL\".split(\"/\")[-1].split(\".\")[0])")


# ============================================================================
# Automatic stuff below


# ==========
# Environmental set-up

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


# ==========
# Set-up SocialIQA

echo -e "\n###############################################################"
echo "# Setting up, convert_socialiqa.py"
echo "###############################################################"

if [ ! -d "$SOCIALIQA" ] ; then
  mkdir "$SOCIALIQA"
fi

# Download the dataset
wget https://maartensap.github.io/social-iqa/data/socialIQa_v1.4.tgz -O "$SOCIALIQA"/socialIQa_v1.4.tgz

# Unzip the dataset
tar -xf "$SOCIALIQA/socialIQa_v1.4.tgz" -C "$SOCIALIQA"

# Convert the dataset to the format expected by the training script
python ./finetuning/convert_socialiqa.py "$SOCIALIQA" "$SOCIALIQA"

# ==========
# Set-up BERT

echo -e "\n###############################################################"
echo "# Setting up the BERT model"
echo "###############################################################"

# Directory for the original BERT
if [ ! -d "$BERT_DIR_ORI" ] ; then
  mkdir "$BERT_DIR_ORI"
fi

# Download the original BERT
if [ ! -f "$BERT_DIR_ORI/$BERT_FILE_NAME.zip" ] ; then
  wget "$BERT_URL" -O "$BERT_DIR_ORI/$BERT_FILE_NAME.zip"
  unzip "$BERT_DIR_ORI/$BERT_FILE_NAME.zip" -d "$BERT_DIR_ORI/"
  mv "$BERT_DIR_ORI/$BERT_FILE_NAME"/* "$BERT_DIR_ORI"
fi

# Copy over my own BERT model
echo -e "\n=========="
echo "Copying pre-trained BERT model"
echo -e "==========\n"

cp -r $BERT_TRAINED_DIR "$BERT_DIR_LOC"



# ==
# Run fine-tuning

echo -e "\n###############################################################"
echo "# Running run_socialiqa.py"
echo "###############################################################"

python -u ./finetuning/run_socialiqa.py \
       --data_dir="$SOCIALIQA"\
       --bert_config_file="$BERT_DIR_ORI/bert_config.json" \
       --vocab_file="$BERT_DIR_ORI/vocab.txt" \
       --output_dir="$OUT_DIR" \
       --init_checkpoint="$BERT_DIR_LOC/$BERT_CHECKPT_NAME" \
       --do_lower_case=False \
       --max_seq_length=128 \
       --do_train=True \
       --do_eval=True \
       --do_predict=False \
       --train_batch_size=8 \
       --eval_batch_size=8 \
       --predict_batch_size=8 \
