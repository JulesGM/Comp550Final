# Standard imports
import argparse
import dataclasses
import json
import logging
import numpy as np
import pathlib
import pickle
from typing import Dict, List, Set, Sequence, Union
import utils

# Third-party imports
import nptyping # Just a library to allow type annotations with Numpy
from sklearn import naive_bayes
import tqdm

SAMPLES_PER_OUTPUT_FILE = 1000

class FilterInferenceBase:
    """Base class of all of the different filters.

    This is for the non-adversarial filters.
    Essentially, filters have a function called `filter` that receive samples, 
    and returns whether or not each sample should be included.

    Each model has a configuration file with information specific to that
    type of filter. The schema of the json file is specific to the Filter child
    class.

    Arguments:

    """
    def __init__(self, model_config_path: utils.PathStr):
        with open(model_config_path) as fin:
            self._model_config = json.load(fin)

    def filter(self, samples: nptyping.Array[np.int]
               ) -> nptyping.Array[np.bool]:
        """ Returns a Numpy array with booleans, telling which samples to use.
        """

        raise NotImplementedError("Pure Abstract Function")


# TODO(julesgm, im-ant, FarzanehAskari): Write other Filter inference classes:
# - Hand written rules [maybe multiple classes]
# - Transformer Encoder
# - LSTM
# - ? Invent one.

class NBCFilter(FilterInferenceBase):
    """Smart filter using a Naive Bayes Classifier.
    
    Expects its Json config file to have:
        - model_pkl_path: Place where we can find the pikle file with the model
          save.

    """
    # TODO(julesgm, im-ant): Should probably be moved to a separate file.
    def __init__(self, model_config_path: utils.PathStr):
        """ Loads the model.
        """
        super().__init__(model_config_path)
        with open(self._model_config["model_pkl_path"], "rb") as fin:
            self._model: naive_bayes.CategoricalNB = pickle.load(fin)


    def filter(self, sample: nptyping.Array[np.int]
    ) -> nptyping.Array[np.bool]:
        # TODO(julesgm, im-ant): We should likely do mini-batches.
        return self._model.predict(sample)


FILTER_MAP = dict(naive_bayes_classifier=NBCFilter,             
                  # hand_written_rules=HandWrittenRulesFilter,
                  # etc.
                )

def main(args: argparse.Namespace):
    utils.print_args(args)
    ###########################################################################
    # 1. Load filter
    ###########################################################################
    logging.info("Loading the filter.")
    filter_ = FILTER_MAP[args.filter_type](args.json_config_path)

    ###########################################################################
    # 2. Filter data & save it
    ###########################################################################
    # The format of the output data may change.
    # We are expecting small arrays of ints in numpy's npz format currently.
    stack = []
    active_output_sample_count = 0
    i = 0
    
    # TODO(julesgm, im-ant): Do this in parallel. 
    # The stack thing is not ideal in parallel; it should be easy to come up with 
    # something else though.
    for file_ in tqdm.tqdm(pathlib.Path(args.input_data_path).glob(f"*.npz")):
        logging.debug(f"Loading file \"{file_}\"")
        samples = np.load(file_, dtype=int)
        logging.debug(f"Loaded file \"{file_}\".")
        
        logging.debug(f"Filtering the samples from a file.")
        new_output_samples = samples[filter_.filter(samples)]
        stack.append(new_output_samples)
        active_output_sample_count += len(new_output_samples)

        if active_output_sample_count >= SAMPLES_PER_OUTPUT_FILE:
            samples: nptyping.Array[np.int] = np.concatenate(stack)
            output_samples = samples[:SAMPLES_PER_OUTPUT_FILE]

            # Awkward addition of a suffix. We are saving to different files.
            logging.debug(f"Saving {i}.")
            np.save(pathlib.Path(str(args.output_data_path_prefix) + f"_{i}"),          
                    output_samples)
            logging.debug(f"Saved {i}.")

            if active_output_sample_count > SAMPLES_PER_OUTPUT_FILE:
                stack = [samples[SAMPLES_PER_OUTPUT_FILE:]]
            else:
                stack = []
    logging.info("Done.")
                


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("filter_type", choices=list(FILTER_MAP.keys()), 
                        help="Which type of filter we are using.")
    parser.add_argument("json_config_path", type=pathlib.Path,
                        help="Path of the model's configuration file.")
    parser.add_argument("input_data_path", type=pathlib.Path,
                        help="Path to the data.")
    parser.add_argument("output_data_path_prefix", type=pathlib.Path,
                        help="Prefix of where to save the data.")
    parser.add_argument("--verbosity", "-v", type=int,                     
                        default=10, help="""
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
    args = parser.parse_args()

    # Logging
    FORMAT = '%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger()
    logger.setLevel(args.verbosity)
    
    main(args)