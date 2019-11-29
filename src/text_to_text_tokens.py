"""Segments the text to BERT tokens, and saves the output.
"""
import pathlib

import utils

import fire
import tqdm
from bert import tokenization


def create_tokenizer(vocab_file, do_lower_case=False):
    return tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    

def main(vocab_path: utils.PathStr, input_path: utils.PathStr, 
         output_path: utils.PathStr, force: bool = False):
    """
    Arguments:
        vocab_path:
            Location of the vocabulary.
        input_path:
            Location of the lines of the original dataset.
        output-path:
            Where to save the segmented tokens.
    """
    # Type checks
    utils.check_type(vocab_path, {pathlib.Path, str})
    utils.check_type(input_path, {pathlib.Path, str})
    utils.check_type(output_path, {pathlib.Path, str})

    # Type coercion
    vocab_path = pathlib.Path(vocab_path)
    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)

    if force or not output_path.exists():
        # Load the BERT tokenizer
        tokenizer = create_tokenizer(str(vocab_path), do_lower_case=False)

        # Open the files
        with open(input_path) as fin, open(output_path, "w") as fout:
            # For each line in the original output text
            num_line_input_path = sum(1 for _ in fin)
            fin.seek(0)
            for line in tqdm.tqdm(fin, total=num_line_input_path):
                # Tokenize with the BERT-tokenizer
                tokens = tokenizer.tokenize(line.strip())
                
                # Write to the output file
                fout.write(" ".join(tokens) + "\n")


if __name__ == "__main__":
    fire.Fire(main)