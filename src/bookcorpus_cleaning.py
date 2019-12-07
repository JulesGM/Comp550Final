# ============================================================================
# For cleaning the bookcorpus output
#
# Reads in all .txt files from --input-dir, tokenize and pre-process the
# plaintext, then write out processed text (one per book) to --output-dir
#
# ============================================================================

import argparse
import glob
import logging
import os
import pathlib
import platform
import re
import time
from typing import Iterable, List

if platform.system() == "Darwin":
    blingfire = None
else:
    import blingfire

try:
    import colored_traceback.auto
except ImportError:
    pass
import colorama
import numpy as np
import spacy
import tqdm

from bert import tokenization
import utils
CHUNK_MAX_LEN = 100000
MODES_NEEDING_BLINGFIRE = {"blingfire", "check"}
MODES_NEEDING_BERT_NATIVE = {"bert-native", "check"}
VALID_MODES = MODES_NEEDING_BLINGFIRE | MODES_NEEDING_BERT_NATIVE

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


def filter_sentences(sentences: List[str]) -> Iterable[str]:
    """
    Function to filter a list of sentences

    :param sentences: list of sentences to be cleaned up
    :return: list of cleaned up sentences
    """

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

        yield sent


def generate_textid_corpus(args: argparse.Namespace) -> None:
    """
    Read raw files (in specified directory), parse and filter, then output
    the Bert token-ids for all files to another directory

    :param args: ArgumentParser-parsed arguments
    :return: None
    """

    if not args.mode in VALID_MODES:
        raise ValueError(f"The argument 'mode' needs to be one of "
                         f"{VALID_MODES}, got {args.mode}.")

    if platform.system() == "Darwin" and args.mode in MODES_NEEDING_BLINGFIRE:
        raise Exception(f"Got a mode requiring Blingfire (mode = {args.mode}), "
                        "yet Blingfire doesn't support Macos.")

    if not blingfire:
        # If we aren't using blingfire, then we must use spacy
        # for sentence segmentation.
        try:
            spacy_model = spacy.load("en_core_web_sm")
        except OSError:
            print()
            print("Exception:")
            print("Didn't find the model for spacy.")
            print("Run 'python -m spacy download en_core_web_sm'")
            exit(-1)

    # Get list of input file paths
    in_list = sorted(glob.glob(os.path.join(args.input_dir, "*.txt")))
    if args.max_number_of_books:
        in_list = in_list[:args.max_number_of_books]

        logging.warning(f"{colorama.Fore.RED}>>> USING A MAX NUMBER OF BOOKS <<<"
                        f"{colorama.Style.RESET_ALL}")

    # Load blingfire textid model
    if args.mode == "blingfire" and platform.system() == "Darwin":
        raise Exception("BlingFire is not compatible with MacOS.")
    
    idtok_model = None
    if blingfire and args.mode in MODES_NEEDING_BLINGFIRE:
        model_path = os.path.join(args.textid_dir, args.base_tok_file)
        utils.check_file_exists(model_path)
        idtok_model = blingfire.load_model(model_path)
    
    utils.check_file_exists(args.vocab_path)
    bert_full_tokenizer = tokenization.FullTokenizer(
        vocab_file=str(args.vocab_path), do_lower_case=False)

    if args.mode == "check":
        with open(args.vocab_path) as fin:
            ids_to_words = fin.read().strip().split("\n")
            words_to_ids = {i: word for i, word in enumerate(ids_to_words)}

    # Iterate through each raw file
    if args.mode != "blingfire":
        print("WARNING: We aren't in a mode that doesn't "
              f"exclusively use Blingfire. Will be slow.\nMode: {args.mode}")

    logging.info(f"Main Loop - {args.mode}")
    for i, in_file_path in enumerate(tqdm.tqdm(in_list)):
        # Generate output file path
        file_basename = os.path.splitext(os.path.basename(in_file_path))[0]
        out_file_path = os.path.join(args.output_dir, file_basename)

        # Read file chunk by chunk
        with open(in_file_path) as in_file:
            # We read the whole file, then cut to CHUNK_MAX_LEN characters long.
            # This seems like a more resistant way to guarantee that we 
            # correctly get full sentences.
            # The length of the chunks at 100k is the longuest that doesn't
            # break spacy's sentence tokenizer.
            logging.debug("Loading a file >")
            file_text = in_file.read().strip()
            if not file_text:
                continue

            logging.debug("< Done loading a file")

            for i in range(len(file_text) // CHUNK_MAX_LEN):
                logging.debug("Chunking. >")
                chunk = file_text[i * CHUNK_MAX_LEN:(i + 1) * CHUNK_MAX_LEN]
                # Get the blingfire-processed sentences from this chunk
                # (NOTE: maybe redundant, look into it maybe removing if slow)
                sent_tok_start = time.time()
                logging.debug("< Done chunking.")

                logging.debug("Segmentizing sentence. >")
                if blingfire:
                    sentences = chunk_to_sentences(chunk)
                else:
                    sentences = [str(x) for x in spacy_model(chunk).sents]
                # Ignore the first and last sentences, as they've
                # likely been cut weirdly by the chunking process.
                # We loose less than 1/1000th of all sentences by doing this.
                # (with a CHUNK_MAX_LEN of 100k).
                logging.debug(f"Number of sentences: {len(sentences)}")
                sentences = sentences[1:-1]

                logging.debug(f"< Done segmentizing sentence. It took "
                            f"{time.time() - sent_tok_start} seconds.")
                # Additional filtering for plaintext sentences
                filter_time_start = time.time()
                logging.debug("Filtering sentences >")
                ft_sentences = filter_sentences(sentences)
                logging.debug(f"< Done filtering sentences. It took "
                            f"{time.time() - filter_time_start} seconds.")

                # Convert each sentence to their textid
                bpe_tok_time_start = time.time()
                logging.debug("Tokenizing sentences >")

                curr_ids = utils.TypedList(np.ndarray)
                for ft_sent in ft_sentences:
                    ids = None
                    if blingfire:
                        ids = blingfire.text_to_ids(idtok_model, ft_sent,
                                                    args.id_seq_length,
                                                    args.oov_id)

                    if args.mode == "bert-native" or args.mode == "check":
                        bert_tokens = bert_full_tokenizer.tokenize(ft_sent)
                        bert_tok_ids = bert_full_tokenizer.convert_tokens_to_ids(
                            bert_tokens)

                        bert_tok_ids_ = utils.TypedList(int)
                        for x in bert_tok_ids:
                            bert_tok_ids_.append(x)
                        bert_tok_ids = bert_tok_ids_
                        
                        while len(bert_tok_ids) < args.id_seq_length:
                            bert_tok_ids.append(0)
                        
                        bert_tok_ids = np.array(list(bert_tok_ids), 
                            dtype=np.int32)[:args.id_seq_length] 

                        if args.mode == "bert-native":
                            ids = bert_tok_ids

                    if args.mode == "check":
                        # In the "check" mode, we test that both the
                        # bert native tokenizer and blingfire return 
                        # the same thing.
                        
                        utils.check_equal(ids.shape, bert_tok_ids.shape)
                        comp = ids == bert_tok_ids
                        
                        if not np.all(comp):
                            def bert_decode(ids):
                                return " ".join(ids_to_words[wid] 
                                    for wid in ids if wid != 0)#.replace(" ##", "")

                            # print("Blingfire ids:")
                            # print(ids)
                            print("\n################################################")
                            print("Mismatch between decoders:")
                            print(f"\t Blingfire decoded: \"{bert_decode(ids)}\"")
                            print(f"\t- Bert-native decoded: \"{bert_decode(bert_tok_ids)}\"")
                            print("################################################\n")
                            # print("Bert-native tokenizer ids:")
                            # print(bert_tok_ids)
                            
                            num_errors = np.sum(np.logical_not(comp))
                            out_of = max(np.sum(ids != 0), 
                                         np.sum(bert_tok_ids != 0))

                            if num_errors/out_of >= 1:
                                raise ValueError(f"{num_errors} "
                                             f"different out of {out_of} "
                                             f"non padding values")

                    curr_ids.append(ids)

                logging.debug(f"< Done tokenizing sentences. It took "
                              f"{time.time() - bpe_tok_time_start} seconds.")
                
                concat_time_start = time.time()
                logging.debug("Concatenating the ids. >")

                if not curr_ids:
                    logging.warning(">> Warning: empty cur_file_ids")

                id_mat = np.array(list(curr_ids), dtype=np.int32)

                logging.debug(f"< Done Concatenating the ids. Took "
                              f"{time.time() - concat_time_start} seconds.")
                if len(id_mat) == 0:
                    logging.warn(f"We got an id_mat of size 0.\nFile index = {i}."
                                f"\nBook file path = {in_file_path}.")
                logging.debug("Saving >")
                path = pathlib.Path(out_file_path)
                np.save(path.parent/(f"{i}_" + str(path.name)), id_mat) 
                logging.debug("< Done saving.")

    # Free model
    if blingfire:
        blingfire.free_model(idtok_model)


if __name__ == "__main__":
    # Parsing input arguments
    parser = argparse.ArgumentParser(description="Clean up set of plaintext books")
    parser.add_argument("--max_number_of_books", "-mnob", type=int,
                        default=None)
    parser.add_argument("--input-dir", type=str, required=True,
                        help="path to input directory to read from (default: cwd)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="path to output directory to write to (default: cwd)")
    parser.add_argument("--textid-dir", type=str,
                        default=(os.path.dirname(blingfire.__file__) 
                        if blingfire else None),
                        help="""path to directory of the text-id file (default: 
                                os.path.dirname(blingfire.__file__))""")
    parser.add_argument("--base-tok-file", type=str, default="bert_base_cased_tok.bin",
                        help="""file name of the base token id file (default: 
                                bert_base_tok.bin)""")
    parser.add_argument("--min-sent-len", type=int, default=4, metavar="N",
                        help="minimum token length of valid sentence (default: 4)")
    parser.add_argument("--remove-blank", type=bool, default=True,
                        help="remove lines with just whitespace (default: True)")
    parser.add_argument("--remove-heads", type=int, default=0,
                        help="remove first N lines of file (default: N=0)")
    parser.add_argument("--id-seq-length", type=int, default=128,
                        help="""sequence length for text id tokenization 
                                (default: 128)""")
    parser.add_argument("--oov-id", type=int, default=100,
                        help="OOV id for text id tokenization (default: N=100)")

    parser.add_argument("--vocab_path", "-rp", type=pathlib.Path,
                        required=True)
    parser.add_argument("--mode", "-m", choices=VALID_MODES,
                        required=True)
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
    # TODO: add verbosity so we know the book being filtered

    args = parser.parse_args()
    # generate_plaintext_corpus(args)
    logging.basicConfig(format='%(message)s')
    utils.log_args(args)
    logging.getLogger().setLevel(args.verbosity)

    """
    TODO:
    - apache spark?    
    - add some form of logging for each book
    """

    generate_textid_corpus(args)
