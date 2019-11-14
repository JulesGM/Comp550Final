set -e # Close immidiately if a line returns something else than 0 (aka, if there is an error)
set -u # Close immidiately if we try to access a variable that doesn't exist.


VOCAB_PATH="vocab.txt"
INPUT_PATH="socialiqa-train-output.txt"
OUTPUT_PATH="splitted-socialiqa.txt"
BERT_LIB_PATH="bert"
VOCAB_URL="https://raw.githubusercontent.com/microsoft/BlingFire/master/ldbsrc/bert_base_cased_tok/vocab.txt"

# Minor compatibility adjustment for the BERT code.
# find "$BERT_LIB_PATH" -iname "*.py" -exec sed -i 's/tf.gfile./tf.io.gfile./g' "{}"  \;

# Download an example BERT vocab if it doesn't exist.
if [ ! -f "$VOCAB_PATH" ] ; then
    echo "Downloading the BERT vocab"
    wget "$VOCAB_URL" -O "$VOCAB_PATH"
fi

# run `extract_socialiqa.py` with the default arguments
echo "Exctacting the lines from the SocialIQA training set."
python extract_socialiqa.py context question

# run the segmentation script.
python text_to_text_tokens.py --vocab_path="$VOCAB_PATH" --input_path="$INPUT_PATH" --output_path="$OUTPUT_PATH"
