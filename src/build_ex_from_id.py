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
import random
import re
from typing import Dict, List

import numpy as np
import tqdm

import utils
import tf_example_utils


def sample_rand_sent(file_paths: List[str], n: int,
                     npy_cache: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Sample a matrix of n random sentences of length l, given the file paths
    of books (in the form of numpy matrices containing the token IDs for
    each sentence in each book)

    :param file_paths: list of of paths to .npy files we sample from
    :param n: number of samples we want to generate
    :param l: sequence length of each sentence (default: 128)
    :return: np.ndarray of sampled sentences (in id)
    """

    count = 0
    bad_files = set()
    # Go over each book and sample
    while count < n:
        if len(bad_files) == len(file_paths):
            raise RuntimeError("All the files are bad")
        # Open random book and sample random sentence indices
        target_file = random.choice(file_paths)
        if target_file in bad_files:
            continue
        if target_file not in npy_cache:
            new_file = np.load(target_file)
            if any(x == 0 for x in new_file.shape):
                logging.info(f"skipped one file - it was empty: {target_file}")
                bad_files.add(target_file)
                continue
            else:
                npy_cache[target_file] = new_file
        cur_book_mat = npy_cache[target_file]
        count += 1

        yield random.choice(cur_book_mat)


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
    logging.debug(" - " + "\n - ".join(in_list))

    # Iterate through each book file
    logging.info("generate_tf_example:")
    npy_cache = {}
    for i, in_file_path in enumerate(tqdm.tqdm(in_list)):
        logging.debug("[%d / %d] : %s" % (i + 1, len(in_list), in_file_path))
        # Load id matrix
        if in_file_path not in npy_cache:
            npy_cache[in_file_path] = np.load(in_file_path)
        id_mat = npy_cache[in_file_path]
        if any(x == 0 for x in id_mat.shape):
            continue

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
        rand_sent_mat = np.array(list(sample_rand_sent(in_list,
                                                       num_rand_next_send,
                                                       npy_cache)))
        rand_sent_mat_idx = 0

        # Add to tf example
        for curr_sent_idx in sent_indices:
            curr_sent = id_mat[curr_sent_idx]
            next_is_rand = curr_sent_idx in send_indices_wrand
            if next_is_rand:
                next_sent = rand_sent_mat[rand_sent_mat_idx]
                rand_sent_mat_idx += 1
            else:
                next_sent = id_mat[curr_sent_idx + 1]

            # Write tf example
            # The != 0 is to remove the padding.
            writer.add_sample(curr_sent[curr_sent != 0],
                              next_sent[next_sent != 0], next_is_rand)


def main(args: argparse.Namespace) -> None:
    """
    Main method for building tf examples from individual book (.npy) files

    :param args: ArgumentParser-parsed arguments
    :return: None
    """
    utils.log_args(args)

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
    parser = argparse.ArgumentParser(description=
        "Clean up set of plaintext books")

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
    parser.add_argument("--num-example-files", type=int, default=1000,
                        help="number of tf example files to generate "
                             "(default: 1000)")
    parser.add_argument("--verbosity", "-v", type=int, 
                            default=int(logging.INFO), help="""
                            Verbosity levels in python: 
                                NOTSET = 0
                                DEBUG = 10
                                INFO = 20
                                WARNING = 30 
                                WARN = WARNING
                                ERROR = 40 
                                CRITICAL = 50 
                                FATAL = CRITICAL         
                            """)
    # TODO maybe: add verbosity so we control knowing the book being filtered?

    args = parser.parse_args()
    logging.getLogger().setLevel(args.verbosity)
    main(args)

