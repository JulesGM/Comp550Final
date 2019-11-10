# ============================================================================
# For cleaning the bookcorpus output
#
# ============================================================================

import os
import glob
import argparse
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




def main():
    #NOTE: not complete function
    # Get list of input file paths
    in_list = list(sorted(glob.glob(os.path.join(args.input_dir, '*.txt'))))

    # Read each file and pre-process
    for i, file_path in enumerate(in_list):
        # Read file, tokenize and generate sentences list
        sents, _ = convert_to_sentences(open(file_path).readlines())

    """
    # Filter

        for j, s in enumerate(sents):
            print(j, s, s.isspace())
            if j>30:
                break

        break
    """


    """
    TODO:
    - min sentence length
    - remove blank sentences?
    - figure out what the above method (blingfire) does
        - can blingfire do all of the above?
    - remove header?
    - apache spark?
    - output one file per book
    
    """

    # now i can write out teha bove as a single text file which has sentences as lines
    #   for the book that i just read in


if __name__ == "__main__":
    main()