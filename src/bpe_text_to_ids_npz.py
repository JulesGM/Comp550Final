import os
import sys
import pathlib

import fire
import numpy as np

import utils


def main(bert_vocab_path: utils.PathStr, input_data_path: utils.PathStr,
         output_path: utils.PathStr):
    """ Converts files in the textual BPE token format to dense bpe ids npz format.

    This is only for a prototype. Converts the text bpe format to the 
    dense numpy matrix format. I'm sure we will transition to a different format
    soon.
    The main advantage over text is that this is in theory faster, as you don't 
    have to parse the text to numbers. The problem is the crazy padding at the 
    end of each sentence, and the fact that you need to load the whole thing to 
    memory at once.
    
    TODO:
    The ideal would be the tf.Example format, because it is a binary format, but
    it doesn't require padding.

    Arguments:
        bert_vocab_path: 
            Path to the vocabulary file used to generate the tokenization.
        input_data_path:
            Path to the data needing to be converted.
        output_path:
            Path to where the npz needs to be saved
    """
    utils.check_type(bert_vocab_path, [pathlib.Path, str])
    utils.check_type(input_data_path, [pathlib.Path, str])

    # Open the vocab file
    with open(bert_vocab_path) as fin:
        # Vocabulary mapping of the BERT id to the token text
        id_to_text = [token.strip() for token in fin]
    # Vocabulary mapping of the token text to their BERT id
    text_to_id = {text: id_ for id_, text in enumerate(id_to_text)}

    ids_per_line = []
    # Open the file with the lines of the dataset
    with open(input_data_path) as fin:
        # For each line of the input file
        for line in fin:
            # Extract the tokens of the line
            tokens_of_the_line = line.strip().split(" ")
            # Convert the tokens to ids
            ids_of_the_line = [text_to_id[token_text.strip()] 
                               for token_text in tokens_of_the_line]
            # Add the list of the ids of the line to a list
            ids_per_line.append(ids_of_the_line)

    # Find the longuest line
    max_l = max(len(line) for line in ids_per_line)
    # Allocate a matrix that number of ints wide
    dense_mat = np.zeros(shape=(len(ids_per_line), max_l), dtype=np.int)
    # Fill in the values of the dense matrix
    for i, sample in enumerate(ids_per_line):
        dense_mat[i, :len(sample)] = sample

    # Save the matrix
    np.save(output_path, dense_mat)


if __name__ == "__main__":
    fire.Fire(main)