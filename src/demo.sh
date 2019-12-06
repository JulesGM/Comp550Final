# The script prepares the socialIQA dataset for the training of 
# the smart filter.
# 1. Downloads the BERT vocab.
# 2. Runs the "extract_socialiqa.py" script to download and extract
#    the lines of the socialIQA dataset.
# 3. Bert segmentizes (as text tokens) the text lines.
# 4. Converts the bert text tokens to tf.Examples 
#    by running 'bpe_text_to_ids_tf_examples.py'

set -e # Close immidiately if a line returns something else than 0 (aka, if there is an error)
set -u # Close immidiately if we try to access a variable that doesn't exist.

DATA_DIR="/network/home/gagnonju/data"
UNLABELED_DIR="$SLURM_TMPDIR/tf_examples_dir/*.tfrecord"

TEXT_LINES="$SLURM_TMPDIR/socialiqa-train-output.txt"
OUTPUT_PATH_TXT_BPE="$SLURM_TMPDIR/splitted-socialiqa.txt"
OUTPUT_PATH_TF_EXAMPLES="$SLURM_TMPDIR/splitted-socialiqa.tfexamples"
BERT_LIB_PATH="bert"
VOCAB_PATH="$SLURM_TMPDIR/vocab.txt"
VOCAB_URL="https://raw.githubusercontent.com/microsoft/BlingFire/master/ldbsrc/bert_base_cased_tok/vocab.txt"
MODEL_SAVE_PATH="$SLURM_TMPDIR/model_nbc.pkl"
MODEL_CONFIG_PATH_INFERENCE="../configs/nbc_inference.json"
MODEL_CONFIG_PATH_TRAINING="../configs/nbc_training.json"
FILTERED_OUTPUT_PATH="$SLURM_TMPDIR/final_output.tfexamples"
NUM_TOKENS=128
FORCE="False" # "True" or "False"

VENV_PATH="$SLURM_TMPDIR/cur_venv"
module load python/3.7
module load python/3.7/tensorflow-gpu/2.0.0
# Set up and activate temporary virtualenv
if [ ! -d "$VENV_PATH" ] ; then
  virtualenv "$VENV_PATH"
fi
source "$VENV_PATH/bin/activate"
python -m pip install numpy scipy pandas tqdm spacy pygments colored_traceback -q
python -m pip install nltk fire scikit-learn -q

# Minor compatibility adjustment for the BERT code.
find "$BERT_LIB_PATH" -iname "*.py" -exec sed -i 's/tf.gfile./tf.io.gfile./g' "{}"  \;

# Download an example BERT vocab if it doesn't exist.
if [[ "$FORCE" == "True" ]] || [ ! -f "$VOCAB_PATH" ] ; then
    echo -e "\n####################################################"
    echo "Downloading the BERT vocab"
    echo "####################################################"
    wget "$VOCAB_URL" -O "$VOCAB_PATH"
fi

# Run `extract_socialiqa.py` with the default arguments
echo -e "\n####################################################"
echo "# extract_socialiqa.py:"
echo "####################################################"
python extract_socialiqa.py context --input_path=/tmp/socialiqa-train-dev/train.jsonl \
        --output_path="$TEXT_LINES" --force="$FORCE"

# Run the segmentation script.
echo -e "\n####################################################"
echo "# text_to_text_tokens.py:"
echo "####################################################"
python text_to_text_tokens.py --vocab_path="$VOCAB_PATH" --force="$FORCE" \
    --input_path="$TEXT_LINES" --output_path="$OUTPUT_PATH_TXT_BPE"

# Transfer the ids to tf.Examples. Right now, uses the labeled dataset twice as
# a proof of concept (as we don't have the unlabeled data).
echo -e "\n####################################################"
echo "# bpe_text_to_ids_tf_examples.py:"
echo "####################################################"
python bpe_text_to_ids_tf_examples.py --bert_vocab_path="$VOCAB_PATH" \
    --input_data_path="$OUTPUT_PATH_TXT_BPE" --force="$FORCE" \
    --output_path="$OUTPUT_PATH_TF_EXAMPLES" --max_num_tokens="$NUM_TOKENS"

# Train the filter. Right now, runs over the labeled flattened dataset, as
# a proof of concept (as we don't have the unlabeled data).
echo -e "\n####################################################"
echo "# Filter Training"
echo "####################################################"
python filter_training.py --glob_prefix_labeled_data="$OUTPUT_PATH_TF_EXAMPLES" \
          --model_config_path="$MODEL_CONFIG_PATH_TRAINING" --model_type=NBC \
          --trainer_save_path="$MODEL_SAVE_PATH" \
          --glob_prefix_unlabeled_data="$UNLABELED_DIR" \
          --verbosity=10 \
          --batch_size=1024 \
         --num_threads_reader=4
        
# Run the filter. Right now, runs over the labeled flattened dataset, as
# a proof of concept (as we don't have the unlabeled data).
echo -e "\n####################################################"
echo "# Filter Inference"
echo "####################################################"
python filter_inference.py --filter_type=nbc --batch_size=10 -v=0 \
        --num_map_threads=4 \
        --shuffle_buffer_size=1000 \
        --output_data_path="$FILTERED_OUTPUT_PATH" \
        --input_data_path="$UNLABELED_DIR" \
        --json_config_path="$MODEL_CONFIG_PATH_INFERENCE" \
        --vocab_path="$VOCAB_PATH" \
        --model_ckpt_path="$MODEL_SAVE_PATH"
