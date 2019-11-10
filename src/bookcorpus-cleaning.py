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


# ===============================================
# Functions

def convert_to_sentences(lines: list) -> (list, int):
    """
    Reads the raw lines from a .txt file, tokenize and return a list of
     sentences. Each each sentence has its own line.
    Code taken from:
     https://github.com/soskek/bookcorpus/blob/master/make_sentlines.py

    :param   lines: list of strings from a single .txt file (i.e. one book)
    :return  sent_L: list of sentences
             n_sent: number of sentences
    """
    stack = []
    sent_L = []
    n_sent = 0
    for chunk in lines:
        if not chunk.strip():
            if stack:
                sents = blingfire.text_to_sentences(
                    " ".join(stack).strip().replace('\n', ' ')).split('\n')
                sent_L.extend(sents)
                n_sent += len(sents)
                sent_L.append('\n')
                stack = []
            continue
        stack.append(chunk.strip())

    if stack:
        sents = blingfire.text_to_sentences(
            " ".join(stack).strip().replace('\n', ' ')).split('\n')
        sent_L.extend(sents)
        n_sent += len(sents)
    return sent_L, n_sent


def cleanup_sentences(sentences: list) -> list:
    """
    Function to clean up a list of sentences

    :param sentences: list of sentences to be cleaned up
    :return: list of cleaned up sentences
    """

    clean_sentences = []

    for i, sent in enumerate(sentences):
        # Removing head filter
        if i < args.remove_heads:
            continue

        # Blank sentence filter
        if args.remove_blank and sent.isspace():
            continue

        # Minimum sentence length filter
        wordcount = len(re.findall(r'\w+', sent))
        if wordcount < args.min_sent_len:
            continue

        clean_sentences.append(sent)

    return clean_sentences


def main():
    # Get list of input file paths
    in_list = list(sorted(glob.glob(os.path.join(args.input_dir, '*.txt'))))

    # Read each file and pre-process
    for i, file_path in enumerate(in_list):
        # Read file, tokenize and generate sentences list
        sents, _ = convert_to_sentences(open(file_path).readlines())

        # Clean up sentences
        clean_sents = cleanup_sentences(sents)

        # Write file to output file TODO
        file_basename = os.path.basename(file_path)
        file_out_path = os.path.join(args.output_dir, file_basename)
        open(file_out_path, "a").write("\n".join(clean_sents))

    """
    TODO:
    - apache spark?    
    - add some form of logging for each book
    """


if __name__ == "__main__":
    main()
