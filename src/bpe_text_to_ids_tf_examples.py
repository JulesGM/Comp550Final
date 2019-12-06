import os
import sys
import pathlib

import fire
import numpy as np
import tqdm

import utils
import tf_example_utils
try:
    import colored_traceback.auto
except ImportError:
    pass


def main(bert_vocab_path: utils.PathStr, input_data_path: utils.PathStr,
         output_path: utils.PathStr, max_num_tokens: int, force: bool = False):
    """ Converts files in the textual BPE token format to the  tf.Example format.

    This converts files with the text tokens to binary blobs with the ids of 
    the BPE tokens in the BERT vocabulary, in the tf.Example format.

    Arguments:
        bert_vocab_path: 
            Path to the vocabulary file used to generate the tokenization.
        input_data_path:
            Path to the data needing to be converted.
        output_path:
            Path to where the npz needs to be saved
        force:
            Whether to still do the transformation even if there is already
            a file with the expected name at `output_path`.
    """
    utils.check_type_one_of(bert_vocab_path, [pathlib.Path, str])
    utils.check_type_one_of(input_data_path, [pathlib.Path, str])
    utils.check_type_one_of(output_path, [pathlib.Path, str])
    utils.check_type_one_of(max_num_tokens, [int])
    bert_vocab_path = pathlib.Path(bert_vocab_path)
    input_data_path = pathlib.Path(input_data_path)
    output_path = pathlib.Path(output_path)

    if force or not output_path.exists():
        # Open the vocab file
        with open(bert_vocab_path) as fin:
            # Vocabulary mapping of the BERT id to the token text
            id_to_text = [token.strip() for token in fin]
        # Vocabulary mapping of the token text to their BERT id
        text_to_id = {text: id_ for id_, text in enumerate(id_to_text)}
        
        # Open the file with the lines of the dataset
        num_lines = utils.count_lines(input_data_path)
        with tqdm.tqdm(open(input_data_path), total=num_lines) as fin, \
            tf_example_utils.WriteAsTfExample(output_files=[output_path], 
            max_num_tokens=max_num_tokens, vocab_path=bert_vocab_path) as writer:
            
            # Iterate by packs of two lines to 
            filtered_file_gen = filter(lambda line: not line.isspace(), fin)
            for two_lines in utils.grouper(2, filtered_file_gen, mode="shortest"):
                ids_per_line = []
                for line in two_lines:
                    # Extract the tokens of the line
                    tokens_of_the_line = line.strip().split(" ")
                    
                    # Convert the tokens to ids
                    ids_of_the_line = [text_to_id[token_text.strip()] 
                                    for token_text in tokens_of_the_line]
                    ids_per_line.append(ids_of_the_line)
                
                # Add the list of the ids of the line to a list
                writer.add_sample(*ids_per_line, b_is_random=False)

if __name__ == "__main__":
    fire.Fire(main)