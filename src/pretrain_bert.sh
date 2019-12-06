#!/bin/bash
#SBATCH --partition=low
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=$SLURM_TMPDIR
#SBATCH --error=$SLURM_TMPDIR

# =============================================================================
# Pre-train BERT
#
# TODO WRITE MORE
# =============================================================================
set -e # Close immidiately if a line returns an error.
set -u # Close immidiately if we try to access a variable that doesn't exist.


# Path variables
PRETRAIN_DATA_DIR="???"                           # location of pretraining data dir
PRETRAIN_OUT_DIR="???"                            # location to put the pre-trained BERT

# BERT-trainin variables
BERT_TRAIN_BATCH_SIZE="32"
BERT_TRAIN_MAX_SEQ_LEN="128"
BERT_TRAIN_MAX_PRED_PER_SEQ="20"
BERT_NUM_TRAIN_STEPS="20"
BERT_NUM_WARMUP_STEPS="10"
BERT_TRAIN_LEARNING_RATE="2e-5"


# Variables for SLURM paths (shouldn't have to change these)
VENV_PATH="$SLURM_TMPDIR/cur_venv"              # path of vitual environment
TRAIN_DATA_LOC="$SLURM_TMPDIR/train_data"       # local copy, pretraining data dir
BERT_BASE_DIR="$SLURM_TMPDIR/bert-dir"          # local directory to put BERT base
PRETRAINING_PY="$SLURM_TMPDIR/run_pretraining.py" # local path of run_pretraining.py
PRETRAIN_OUT_LOC="$SLURM_TMPDIR/model_out"      # local directory for pretrained BERT


# ==
# Set up environment
module load python/3.7

# Create temporary virtual environment and activate
if [ ! -d "$VENV_PATH" ] ; then
  virtualenv $VENV_PATH
fi
source $VENV_PATH/bin/activate

# Get the packages according to requirement.txt
python -m pip install "tensorflow >= 1.11.0"
python -m pip install "tensorflow-gpu  >= 1.11.0"


# ==
# Download the pre-training script and requirements
wget --output-document=$PRETRAINING_PY \
     https://raw.githubusercontent.com/google-research/bert/master/run_pretraining.py


# ==
# Get BERT base cased model if not present
if [ ! -d "$BERT_BASE_DIR" ] ; then
  # Download bert base model
  wget --output-document="$SLURM_TMPDIR/bertcased.zip" \
     https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip

  # Unzip into Bert base model
  unzip "$SLURM_TMPDIR/bertcased.zip" -d "$SLURM_TMPDIR/bertcased-unzipped"

  # Copy just the cased model directory (kind of hacky)
  cp -r "$SLURM_TMPDIR/bertcased-unzipped/cased_L-12_H-768_A-12" $BERT_BASE_DIR
fi


# ==
# Copy the tf-example pretraining data to local directory

if [ ! -d "$TRAIN_DATA_LOC" ] ; then
  cp -r $PRETRAIN_DATA_DIR $TRAIN_DATA_LOC
fi


# ==
# Run BERT pre-training

python -u $PRETRAINING_PY \
          --input_file="$TRAIN_DATA_LOC/*.tfrecord" \
          --output_dir=$PRETRAIN_OUT_LOC \
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


# ===
# Copy the pretrained model directory
# cp -r $PRETRAIN_OUT_LOC $PRETRAIN_OUT_DIR

# TODO: maybe also use a grace period so copying is done when job is killed
