# ============================================================================
# For cleaning the bookcorpus output
#
# Reads in all .txt files from --input-dir, tokenize and pre-process the
# plaintext, then write out processed text (one per book) to --output-dir
#
# ============================================================================

import os
import glob
import argparse
import re
import blingfire


# ===============================================
# Functions


def chunk_to_sentences(chunk: str) -> list:
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


def filter_sentences(sentences: list) -> list:
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
        wordcount = len(re.findall(r'\w+', sent))
        if wordcount < args.min_sent_len:
            continue

        clean_sentences.append(sent)

    return clean_sentences


def main(args: argparse.Namespace):
    # Get list of input file paths
    in_list = sorted(glob.glob(os.path.join(args.input_dir, "*.txt")))

    # Iterate through each raw file
    for i, in_file_path in enumerate(in_list):
        # Generate output file path
        file_basename = os.path.basename(in_file_path)
        out_file_path = os.path.join(args.output_dir, file_basename)

        # Open input and output files
        in_file = open(in_file_path, mode="r")
        out_file = open(out_file_path, mode="a")

        # Iteratively read input file to process
        for chunk in in_file:
            # Get the blingfire-processed sentences from this chunk
            bf_sentences = chunk_to_sentences(chunk)

            # Additional filtering for the sentences
            ft_sentences = filter_sentences(bf_sentences)

            # Write filtered sentences to output file
            for ft_sent in ft_sentences:
                out_file.write("%s\n" % ft_sent)

        # Close input and output files
        in_file.close()
        out_file.close()

    """
    TODO:
    - apache spark?    
    - add some form of logging for each book
    """


if __name__ == "__main__":
    # Parsing input arguments
    parser = argparse.ArgumentParser(description="Clean up set of plaintext books")

    parser.add_argument('--input-dir', type=str, required=True,
                        help='path to input directory to read from (default: cwd)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='path to output directory to write to (default: cwd)')
    parser.add_argument('--min-sent-len', type=int, default=4, metavar='N',
                        help='minimum token length of valid sentence (default: 4)')
    parser.add_argument('--remove-blank', type=bool, default=True,
                        help='remove lines with just whitespace (default: True)')
    parser.add_argument('--remove-heads', type=int, default=0,
                        help='remove first N lines of file (default: N=0)')

    args = parser.parse_args()

    main(args)
