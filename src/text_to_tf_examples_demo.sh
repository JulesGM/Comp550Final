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


VOCAB_PATH="/tmp/vocab.txt"
TEXT_LINES="/tmp/socialiqa-train-output.txt"
OUTPUT_PATH_TXT_BPE="/tmp/splitted-socialiqa.txt"
OUTPUT_PATH_TF_EXAMPLES="/tmp/splitted-socialiqa.example"
BERT_LIB_PATH="bert"
VOCAB_URL="https://raw.githubusercontent.com/microsoft/BlingFire/master/ldbsrc/bert_base_cased_tok/vocab.txt"
MODEL_SAVE_PATH="/tmp/model.pkl"
MODEL_CONFIG_PATH="../configs/nbc.json"

# Minor compatibility adjustment for the BERT code.
find "$BERT_LIB_PATH" -iname "*.py" -exec sed -i 's/tf.gfile./tf.io.gfile./g' "{}"  \;

# Download an example BERT vocab if it doesn't exist.
if [ ! -f "$VOCAB_PATH" ] ; then
    echo "####################################################"
    echo "Downloading the BERT vocab"
    echo "####################################################"
    wget "$VOCAB_URL" -O "$VOCAB_PATH"
fi

# run `extract_socialiqa.py` with the default arguments
echo -e "\n####################################################"
echo "# extract_socialiqa.py:"
echo "####################################################"
python extract_socialiqa.py context --input_path=/tmp/socialiqa-train-dev/train.jsonl \
        --output_path="$TEXT_LINES" 

# run the segmentation script.
echo -e "\n####################################################"
echo "# text_to_text_tokens.py:"
echo "####################################################"
python text_to_text_tokens.py --vocab_path="$VOCAB_PATH" \
    --input_path="$TEXT_LINES" --output_path="$OUTPUT_PATH_TXT_BPE"

# transfer the ids to tf.Examples
echo -e "\n####################################################"
echo "# bpe_text_to_ids_tf_examples.py:"
echo "####################################################"
python bpe_text_to_ids_tf_examples.py --bert_vocab_path="$VOCAB_PATH" \
    --input_data_path="$OUTPUT_PATH_TXT_BPE" \
    --output_path="$OUTPUT_PATH_TF_EXAMPLES"

#
echo -e "\n####################################################"
echo "# Filter Training"
echo "####################################################"
python filter_training.py --flattened_labeled_data_path="$OUTPUT_PATH_TF_EXAMPLES" \
         --model_config_path="$MODEL_CONFIG_PATH" --model_type=NBC \
         --trainer_save_path="$MODEL_SAVE_PATH" \
         --unlabeled_dataset_path="$OUTPUT_PATH_TF_EXAMPLES" \
         --verbosity=10