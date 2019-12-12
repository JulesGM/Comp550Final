#!/bin/bash
# =============================================================================
# Pre-train BERT given a directory of tensorflow example (.tfrecord) files.
# Outputs the pretrained model to a directory of choice.
#
# Assume we will be running this from Comp550Final/src/
#
# =============================================================================
set -e # Close immidiately if a line returns an error.
set -u # Close immidiately if we try to access a variable that doesn't exist.


# Path variables
PRETRAIN_DATA_DIR=$data_dir  # location of pretraining data dir
PRETRAINING_PY="./run_pretraining.py"    # location of the run_pretraining file
PRETRAIN_OUT_DIR=$out_dir    # location to put pre-trained BERT
PRETRAIN_OUT_LOC="$PRETRAIN_OUT_DIR"      # local directory for pretrained BERT (e.g. $SLURM_TMPDIR/model_out")


# BERT-trainin variables
BERT_TRAIN_BATCH_SIZE="32"          # originally 32
BERT_TRAIN_MAX_SEQ_LEN="128"
BERT_TRAIN_MAX_PRED_PER_SEQ="20"
BERT_NUM_TRAIN_STEPS="1000000000"   # num training steps, orig is 20, increase for actual pretraining
BERT_NUM_WARMUP_STEPS="10"
BERT_TRAIN_LEARNING_RATE="2e-5"
SAVE_CHECKPOINTS_STEPS="1500"       # 1600 steps is approx. 10 minutes of training on good GPU
USE_TPU="False"                     # no TPU to use yet


# Variables for SLURM paths (shouldn't have to change these)
VENV_PATH="$SLURM_TMPDIR/cur_venv"              # path of vitual environment
TRAIN_DATA_LOC="$SLURM_TMPDIR/train_data"       # local copy, pretraining data dir
BERT_BASE_DIR="$SLURM_TMPDIR/bert-dir"          # local directory to put BERT base

# ==
# Check GPU allocation
nvidia-smi

# ==
# Set up environment
echo "Setting up environment: $(date)"
module purge
module refresh
module load python/3.7
module load cuda/10.1
module load cuda/10.1/cudnn/7.6
module load python/3.7/tensorflow-gpu/1.15.0rc2

# Create temporary virtual environment and activate
if [ ! -d "$VENV_PATH" ] ; then
  virtualenv $VENV_PATH
fi
source $VENV_PATH/bin/activate

# Get the packages according to requirement.txt
#python -m pip install "tensorflow >= 1.11.0" # NOTE no need for this line, redundant
#python -m pip install "tensorflow-gpu==1.15.0rc2" # TODO only use this for actual GPU


# ==
# Get BERT base cased model if not present
if [ ! -d "$BERT_BASE_DIR" ] ; then
  echo "Downloading BERT Cased: $(date)"

  # Download bert base model
  wget --output-document="$SLURM_TMPDIR/bertcased.zip" \
     https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip

  # Unzip into Bert base model
  unzip "$SLURM_TMPDIR/bertcased.zip" -d "$SLURM_TMPDIR/bertcased-unzipped"

  # Copy just the cased model directory (kind of hacky)
  cp -r "$SLURM_TMPDIR/bertcased-unzipped/cased_L-12_H-768_A-12" "$BERT_BASE_DIR"
fi


# ==
# Copy the tf-example pretraining data to local directory
echo -e "\n=========="
echo "Setting up pretraining data: $(date)"
echo -e "==========\n"
rm -rf TRAIN_DATA_LOC || true
if [ ! -d "$TRAIN_DATA_LOC" ] ; then
  cp -r "$PRETRAIN_DATA_DIR" "$TRAIN_DATA_LOC"
fi



echo -e "\n=========="
echo "Starting BERT pre-training on: : $(date)"
echo -e "==========\n"


python -u "$PRETRAINING_PY" \
          --input_file="$TRAIN_DATA_LOC/*.tfrecord" \
          --output_dir="$PRETRAIN_OUT_LOC" \
          --do_train=True \
          --do_eval=True \
          --bert_config_file="$BERT_BASE_DIR/bert_config.json" \
          --init_checkpoint="$BERT_BASE_DIR/bert_model.ckpt" \
          --train_batch_size=$BERT_TRAIN_BATCH_SIZE \
          --max_seq_length=$BERT_TRAIN_MAX_SEQ_LEN \
          --max_predictions_per_seq=$BERT_TRAIN_MAX_PRED_PER_SEQ \
          --num_train_steps=$BERT_NUM_TRAIN_STEPS \
          --num_warmup_steps=$BERT_NUM_WARMUP_STEPS \
          --learning_rate=$BERT_TRAIN_LEARNING_RATE \
          --save_checkpoints_steps=$SAVE_CHECKPOINTS_STEPS \
          --use_tpu=$USE_TPU \


# ===
# Copy the pretrained model directory (no need, model written to public node directly)
#cp -r $PRETRAIN_OUT_LOC $PRETRAIN_OUT_DIR

# TODO: maybe also use a grace period so copying is done when job is killed


# =============================================================================
# unusued sbatch arguments (now passed from input)
# SBATCH.. --partition=long
# --gres=gpu:volta:1
# --cpus-per-task=2
# --mem=32G
# --time=24:00:00
# --output=/network/tmp1/chenant/sharing/comp-550/bert-pretrain/dec-7_test/output-pretrain.txt
# --error=/network/tmp1/chenant/sharing/comp-550/bert-pretrain/dec-7_test/error-pretrain.txt