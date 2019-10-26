""" Extracts the lines from the SocialIQA training set. 

"""

import argparse
import json
import logging
import utils

URL = "https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip"
FILENAME = "socialiqa-train-dev/train.jsonl"
OUTPUT = "socialiqa-train-output.txt"

def main(args: argparse.Namespace):
    args.fields = set(args.fields)
    utils.maybe_download_and_unzip(URL)

    with open(FILENAME) as fin, open(OUTPUT, "w") as fout:
        for line in fin:
            entry = json.loads(line)
            for k, v in entry.items():
                if args.fields:
                    if not k in args.fields:
                        continue
                v = v.casefold()
                if v[-1] not in ".!?":
                    v += "."
            
                if not args.quiet:
                    print(f"{k}: {v}")
            
                fout.write(v + "\n")

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("fields", metavar="N", nargs="*", 
                        help="Indicate which field of the jsonl file to include"  
                              "output text file. Defaults to taking all fields "
                              "in the.")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    # Logging
    FORMAT = '%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    main(args)