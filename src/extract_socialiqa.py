""" Extracts the lines from the SocialIQA training set. 
Usage:
    python 
"""

import argparse
import pathlib
import json
import logging
import utils

URL = "https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip"
TMP = pathlib.Path("/tmp/")

def main(args: argparse.Namespace):
    args.fields = set(args.fields)
    utils.maybe_download_and_unzip(URL, TMP)

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
    parser.add_argument("--input_path", "-ip", 
                        default=pathlib.Path("/tmp/socialiqa-train-dev/train.jsonl"),
                        type=pathlib.Path)
    parser.add_argument("--output_path", "-op", 
                        default=pathlib.Path("socialiqa-train-output.txt"), type=pathlib.Path)                      
    args = parser.parse_args()

    # Logging
    FORMAT = '%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger()
    logger.setLevel(args.verbosity)
    

    main(args)