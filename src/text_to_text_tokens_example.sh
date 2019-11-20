set -e # Close immidiately if a line returns something else than 0 (aka, if there is an error)
set -u # Close immidiately if we try to access a variable that doesn't exist.


VOCAB_PATH="vocab.txt"
INPUT_PATH="socialiqa-train-output.txt"
OUTPUT_PATH_TXT_BPE="splitted-socialiqa.txt"
OUTPUT_PATH_TF_EXAMPLES="splitted-socialiqa.example"
BERT_LIB_PATH="bert"
VOCAB_URL="https://raw.githubusercontent.com/microsoft/BlingFire/master/ldbsrc/bert_base_cased_tok/vocab.txt"


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
python extract_socialiqa.py context

# run the segmentation script.
echo -e "\n####################################################"
echo "# text_to_text_tokens.py:"
echo "####################################################"
python text_to_text_tokens.py --vocab_path="$VOCAB_PATH" \
    --input_path="$INPUT_PATH" --output_path="$OUTPUT_PATH_TXT_BPE"

# transfer the ids to tf.Examples
echo -e "\n####################################################"
echo "# bpe_text_to_ids_tf_examples.py:"
echo "####################################################"
python bpe_text_to_ids_tf_examples.py --bert_vocab_path="$VOCAB_PATH" \
    --input_data_path="$OUTPUT_PATH_TXT_BPE" \
    --output_path="$OUTPUT_PATH_TF_EXAMPLES"