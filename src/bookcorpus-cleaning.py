# ============================================================================
# For cleaning the bookcorpus output
#
# Reads in all .txt files from --input-dir, tokenize and pre-process the
# plaintext, then write out processed text (one per book) to --output-dir
#
# ============================================================================

import argparse
import glob
import os
import re
from typing import List

import blingfire
import numpy as np


# ===============================================
# Functions


def chunk_to_sentences(chunk: str) -> List[str]:
    """
    Takes a chunk of text from the file object, tokenize via BlingFire and
    return the resulting sentences.

    TODO: why is the Bookcorpus equivalent function so convoluted? See:
    https://github.com/soskek/bookcorpus/blob/master/make_sentlines.py

    :param chunk: chunk of input file, separated by the python open iterator
    :return: list of sentences from chunk
    """

    sentences = blingfire.text_to_sentences(
        chunk.strip().replace("\n", " ")).split("\n")

    return sentences


def filter_sentences(sentences: List[str]) -> List[str]:
    """
    Function to filter a list of sentences

    :param sentences: list of sentences to be cleaned up
    :return: list of cleaned up sentences
    """

    clean_sentences = []

    for i, sent in enumerate(sentences):
        # Removing head filter
        if i < args.remove_heads:
            continue

        # Empty sentence filter
        if args.remove_blank and not sent:
            continue
        # Whitespace sentence filter
        if args.remove_blank and sent.isspace():
            continue

        # Minimum sentence length filter
        wordcount = len(re.findall(r"\w+", sent))
        if wordcount < args.min_sent_len:
            continue

        clean_sentences.append(sent)

    return clean_sentences


def generate_textid_corpus(args: argparse.Namespace) -> None:
    """
    Read raw files (in specified directory), parse and filter, then output
    the Bert token-ids for all files to another directory

    :param args: ArgumentParser-parsed arguments
    :return: None
    """

    # Get list of input file paths
    in_list = sorted(glob.glob(os.path.join(args.input_dir, "*.txt")))

    # Load blingfire textid model
    idtok_model = blingfire.load_model(
        os.path.join(args.textid_dir, args.base_tok_file))

    # Iterate through each raw file
    for i, in_file_path in enumerate(in_list):
        # Generate output file path
        file_basename = os.path.splitext(os.path.basename(in_file_path))[0]
        out_file_path = os.path.join(args.output_dir, file_basename)

        # List to store all sentence id (vectors)
        cur_file_ids = []

        # Read file chunk by chunk
        with open(in_file_path) as in_file:
            for chunk in in_file:
                # Get the blingfire-processed sentences from this chunk
                # (NOTE: maybe redundant, look into it maybe removing if slow)
                bf_sentences = chunk_to_sentences(chunk)

                # Additional filtering for plaintext sentences
                ft_sentences = filter_sentences(bf_sentences)

                # Convert each sentence to their textid
                for ft_sent in ft_sentences:
                    ids = blingfire.text_to_ids(idtok_model, ft_sent,
                                                args.id_seq_length,
                                                args.oov_id)
                    cur_file_ids.append(ids)


        # Save the token ids for this entire file
        id_mat = np.array(cur_file_ids, dtype=np.int32)
        np.save(out_file_path, id_mat)  # TODO: test to ensure it works

    # Free model
    blingfire.free_model(idtok_model)


if __name__ == "__main__":
    # Parsing input arguments
    parser = argparse.ArgumentParser(description="Clean up set of plaintext books")

    parser.add_argument('--input-dir', type=str, required=True,
                        help='path to input directory to read from (default: cwd)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='path to output directory to write to (default: cwd)')
    parser.add_argument('--textid-dir', type=str,
                        default=os.path.dirname(blingfire.__file__),
                        help="""path to directory of the text-id file (default: 
                                os.path.dirname(blingfire.__file__))""")
    parser.add_argument('--base-tok-file', type=str, default='bert_base_tok.bin',
                        help="""file name of the base token id file (default: 
                                bert_base_tok.bin)""")
    parser.add_argument('--min-sent-len', type=int, default=4, metavar='N',
                        help='minimum token length of valid sentence (default: 4)')
    parser.add_argument('--remove-blank', type=bool, default=True,
                        help='remove lines with just whitespace (default: True)')
    parser.add_argument('--remove-heads', type=int, default=0,
                        help='remove first N lines of file (default: N=0)')
    parser.add_argument('--id-seq-length', type=int, default=128,
                        help="""sequence length for text id tokenization 
                                (default: 128)""")
    parser.add_argument('--oov-id', type=int, default=100,
                        help='OOV id for text id tokenization (default: N=100)')

    # TODO: add verbosity so we know the book being filtered

    args = parser.parse_args()
    print(args)

    """
    TODO:
    - apache spark?    
    - add some form of logging for each book
    """

    # generate_plaintext_corpus(args)
    generate_textid_corpus(args)
