# ============================================================================
# Generate tensorflow examples from a set of .npy files, each containing the
# token-id to generate tf example for classifier training.
#
# Specifically, each "book file" is a numpy matrix (.npy file) where each row
# is one sentence where each word is represented by their token id.
#
# Reads in all .npy files from --input-dir to generate tf example
#
#
# Potential TODOs:
#   - apache spark?
#   - add some form of logging for each book?
#
# ============================================================================

import argparse
from collections import Counter
import glob
import logging
import os
import pathlib
import re
from typing import List

import blingfire
import numpy as np

import tf_example_utils


def sample_rand_sent(file_paths: List[str], n: int, l: int = 128
                     ) -> np.ndarray:
    """
    Sample a matrix of n random sentences of length l, given the file paths
    of books (in the form of numpy matrices containing the token IDs for
    each sentence in each book)

    :param file_paths: list of of paths to .npy files we sample from
    :param n: number of samples we want to generate
    :param l: sequence length of each sentence (default: 128)
    :return: np.ndarray of sampled sentences (in id)
    """

    # Sample n random books indices
    sampl_book_idxs = np.random.choice(len(file_paths), size=n, replace=True)

    # Initialize output matrix and write order
    write_order = np.random.choice(n, size=n, replace=False)
    write_order_idx = 0
    rand_sent_mat = np.empty((n, l), dtype=np.uint32)

    # Dictionary mapping: file index -> number of samples from the file
    bookidx_2_numsent = dict(Counter(sampl_book_idxs))

    # Go over each book and sample
    for book_idx in bookidx_2_numsent:
        # Open random book and sample random sentence indices
        cur_book_mat = np.load(file_paths[book_idx])
        sent_idxs = np.random.choice(len(cur_book_mat),
                                     size=bookidx_2_numsent[book_idx],
                                     replace=True)
        # Get sentences and save to matrix
        for j in sent_idxs:
            rand_sent_mat[write_order_idx] = cur_book_mat[j]
            write_order_idx += 1

    return rand_sent_mat


def generate_tf_example(args: argparse.Namespace,
                        writer: tf_example_utils.WriteAsTfExample) -> None:
    """
    Generate tf examples for BERT pre-training from a corpus of numpy matrices
    denoting the token IDs for each token from some bookcorpus

    :param args: ArgumentParser-parsed arguments
    :param writer: WriteAsTfExample object for writing tf examples to files
    :return: None
    """

    # Get list of all available books
    in_list = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")))

    # TODO maybe: pretty-fy using progress bars?

    # Iterate through each book file
    for i, in_file_path in enumerate(in_list):
        print("[%d/%d] %s" % (i+1, len(in_list), in_file_path))
        # Load id matrix
        id_mat = np.load(in_file_path)
        logging.debug(in_file_path)
        logging.debug(id_mat.shape)
        if len(id_mat) == 0:
            logging.warn(f"Got an id_mat of size 0. Path: {in_file_path}")

        # Figure out how many rows do we want to sample from each book
        num_to_sample = (len(id_mat) - 1 if args.sent_per_book == -1
                         else args.sent_per_book)
        # If we want to sample a list of shuffled row indices
        if args.shuf_sentences:
            sent_indices = np.random.choice(len(id_mat) - 1,
                                            size=min(num_to_sample,
                                                     len(id_mat) - 1),
                                            replace=False)
        # If we just want to take row indices in order
        else:
            sent_indices = np.arange(0, stop=min(num_to_sample,
                                                 len(id_mat) - 1))

        # Pre-specify the sentence indices to have a random next sentence
        num_rand_next_send = int(args.rand_sent_prob * len(sent_indices))
        send_indices_wrand = np.random.choice(sent_indices,
                                              size=num_rand_next_send,
                                              replace=False)
        # Pre-sample the random sentence to follow
        rand_sent_mat = sample_rand_sent(in_list, num_rand_next_send, l=128)
        rand_sent_mat_idx = 0

        # Add to tf example
        for cursent_idx in sent_indices:
            cur_sent = id_mat[cursent_idx]
            next_is_rand = cursent_idx in send_indices_wrand
            if next_is_rand:
                next_sent = rand_sent_mat[rand_sent_mat_idx]
                rand_sent_mat_idx += 1
            else:
                next_sent = id_mat[cursent_idx + 1]

            # Write tf example
            writer.add_sample(cur_sent, next_sent, next_is_rand)


def main(args: argparse.Namespace) -> None:
    """
    Main method for building tf examples from individual book (.npy) files

    :param args: ArgumentParser-parsed arguments
    :return: None
    """

    # Initialize the list of output files to write the examples to
    output_files = []
    for i_tf_ex in range(args.num_example_files):
        cur_tf_file_name = "%d_TfExample.tfrecord" % i_tf_ex
        output_files.append(os.path.join(args.output_dir, cur_tf_file_name))

    # Generate examples
    with tf_example_utils.WriteAsTfExample(output_files, args.vocab_file,
                                        args.max_num_tokens) as writer:
        generate_tf_example(args, writer)


if __name__ == "__main__":
    # Parsing input arguments
    parser = argparse.ArgumentParser(description="Clean up set of plaintext books")

    parser.add_argument("--input-dir", type=str, required=True,
                        help="path to input directory to read from")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="path to output directory to write to")
    parser.add_argument("--vocab-file", type=str, required=True,
                        help="path to the BERT vocabulary file")
    parser.add_argument("--shuf-sentences", type=bool, default=False,
                        help="""whether to randomize the sentences from each 
                                book (default: False)""")
    parser.add_argument("--sent-per-book", type=int, default=-1,
                        help="""number of sentence examples to sample from per 
                                book (default: -1, for all sentences)""")
    parser.add_argument("--rand-sent-prob", type=float, default=0.5,
                        help="""probability of sampling a random next sentence 
                                (default: 0.5)""")
    parser.add_argument("--max-num-tokens", type=int, default=128,
                        help="""maximum allowable example sentence length,
                                counted as number of tokens (default: 128)""")
    parser.add_argument("--num-example-files", type=int, default=3,
                        help="number of tf example files to generate (default: 3)")

    # TODO maybe: add verbosity so we control knowing the book being filtered?

    args = parser.parse_args()
    print(args)
    logging.getLogger().setLevel(logging.DEBUG)

    main(args)

