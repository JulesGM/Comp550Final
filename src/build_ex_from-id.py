# ============================================================================
# Generate tensorflow examples from a set of files, each containing token-id
# to generate tf example for classifier training.
#
#
# Reads in all .npy files from --input-dir to generate tf example
#
# ============================================================================

import argparse
from collections import Counter
import glob
import os
import pathlib
import re
from typing import List

import blingfire
import numpy as np

import to_tf_example


# ===============================================
# Functions


def sample_rand_sent(file_paths: List[str], n: int, l: int = 128) \
        -> np.ndarray:
    """
    TODO: write ths

    :param file_paths: list of of paths to .npy files we sample from
    :param n: number of samples we want to generate
    :param l: sequence length of each sentence (default: 128)
    :return: np.ndarray of sampled sentences (in id)
    """

    # Sample n random books indeces
    sampl_book_idxs = np.random.choice(len(file_paths), size=n, replace=True)

    # Initialize output matrix and write order
    write_order = np.random.choice(n, size=n, replace=False)
    write_order_idx = 0
    rand_sent_mat = np.empty((n, l), dtype=np.uint32)

    # Dictionary mapping: file index -> number of samples from the file
    bookidx_2_numsent = dict(Counter(sampl_book_idxs))

    # Go over each book and sample
    for book_idx in bookidx_2_numsent:
        # Open random book and sample random sentence indeces
        cur_book_mat = np.load(file_paths[book_idx])
        sent_idxs = np.random.choice(len(cur_book_mat),
                                     size=bookidx_2_numsent[book_idx],
                                     replace=True)
        # Get sentences and save to matrix
        for j in sent_idxs:
            rand_sent_mat[write_order_idx] = cur_book_mat[j]
            write_order_idx += 1

    return rand_sent_mat


def generate_tf_example(args, writer):
    """
    TODO: write this

    :param args:
    :return:
    """

    # Get list of all available books
    in_list = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")))

    # TODO: pretty-fy using progress bars?

    # Iterate through each book file
    for i, in_file_path in enumerate(in_list):
        # Load id matrix
        id_mat = np.load(in_file_path)

        # Figure out how many rows do we want to sample from each book
        num_to_sample = len(id_mat) - 1 if args.sent_per_book == -1 \
            else args.sent_per_book
        # If we want to sample a list of shufffled row indeces
        if args.shuf_sentences:
            sent_indeces = np.random.choice(len(id_mat) - 1,
                                            size=min(num_to_sample,
                                                     len(id_mat) - 1),
                                            replace=False)
        # If we just want to take row indeces in order
        else:
            sent_indeces = np.arange(0, stop=min(num_to_sample,
                                                 len(id_mat) - 1))

        # Specify the sentence indeces to have a random next sentence
        num_rand_next_send = int(args.rand_sent_prob * len(sent_indeces))
        send_indeces_wrand = np.random.choice(sent_indeces,
                                              size=num_rand_next_send,
                                              replace=False)
        # Pre-sample the random sentence to follow
        rand_sent_mat = sample_rand_sent(in_list, num_rand_next_send, l=128)
        rand_sent_mat_idx = 0

        # Add to tf example
        for cursent_idx in sent_indeces:
            cur_sent = id_mat[cursent_idx]
            next_is_rand = cursent_idx in send_indeces_wrand
            if next_is_rand:
                next_sent = rand_sent_mat[rand_sent_mat_idx]
                rand_sent_mat_idx += 1
            else:
                next_sent = id_mat[cursent_idx + 1]

            # Add to tf example TODO make sure this works
            writer.add_sample(cur_sent, next_sent, next_is_rand)

        print(np.shape(sent_indeces), np.shape(rand_sent_mat))  # TODO delete
        if i > 3:
            break  # TODO; deelete


if __name__ == "__main__":
    # Parsing input arguments
    parser = argparse.ArgumentParser(description="Clean up set of plaintext books")

    parser.add_argument('--input-dir', type=str, required=True,
                        help='path to input directory to read from')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='path to output directory to write to')
    parser.add_argument('--vocab-file', type=str, required=True,
                        help='path to the BERT vocabulary file')
    parser.add_argument('--shuf-sentences', type=bool, default=False,
                        help="""whether to randomize the sentences from each 
                                book (default: False)""")
    parser.add_argument('--sent-per-book', type=int, default=-1,
                        help="""number of sentence examples to sample from per 
                                book (default: -1, for all sentences)""")
    parser.add_argument('--rand-sent-prob', type=float, default=0.5,
                        help="""probability of sampling a random next sentence 
                                (default: 0.5)""")
    parser.add_argument('--max-num-tokens', type=int, default=128,
                        help="""maximum allowable example sentence length,
                                counted as number of tokens (default: 128)""")

    # TODO: add verbosity so we know the book being filtered

    args = parser.parse_args()
    print(args)

    # Initialize the list of output files to write the examples to
    tmp_tf_file = os.path.join(args.output_dir, 'tf_example.tfrecord')  # TODO: add actual stuff here
    output_files = [tmp_tf_file]  # TODO make better

    # Generate examples
    with to_tf_example.WriteAsTfExample(output_files, args.vocab_file,
                                        args.max_num_tokens) as writer:
        generate_tf_example(args, writer)

    """
        TODO:
        - apache spark?
        - add some form of logging for each book?
    """
