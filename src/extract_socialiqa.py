""" Extracts the lines from the SocialIQA training set. 

Downloads the dataset if it doesn't already exist, unzips it if it needs to,
then extracts the lines.

Usage:
    python src/extract_socialiqa.py 
"""

import argparse
import json
import logging
import pathlib
import utils

URL = "https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip"

def main(args: argparse.Namespace):
    args.fields = set(args.fields)
    utils.maybe_download_and_unzip(URL, args.tmp_dir, force=args.force)

    with open(args.input_path) as fin, open(args.output_path, "w") as fout:
        for line in fin:
            entry = json.loads(line)
            for k, v in entry.items():
                if args.fields:
                    if not k in args.fields:
                        continue
    
                logging.debug(f"{k}: {v}")
            
                # Don't do any preprocessing on v. This will be handled
                # elsewhere to allow for easier uniformity.
                fout.write(v + "\n")


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("fields", metavar="N", nargs="*", 
                        help="Indicate which field of the jsonl file to include"  
                              "output text file. Defaults to taking all fields "
                              "in the.")
    parser.add_argument("--verbosity", "-v", type=int, default=20, help="""
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
    parser.add_argument("--input_path", "-ip", type=pathlib.Path,
                        help="Location of the jsonl with the training examples.")
    parser.add_argument("--output_path", "-op", type=pathlib.Path,
                        help="Where the final text file is saved.")
    parser.add_argument("--tmp_dir", "-td", type=pathlib.Path,
                        help="Where the zip is downloaded and unzipped.")
    parser.add_argument("--force", "-f", type=utils.safe_bool_arg,
                        help="Force redownloading & reunzipping")
    args = parser.parse_args()

    # Logging
    FORMAT = '%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger()
    logger.setLevel(args.verbosity)
    
    main(args)