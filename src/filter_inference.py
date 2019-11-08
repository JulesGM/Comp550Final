"""Filters the unlabeled dataset.
1. Loads the model from a json config file,
2. In a loop:
    a) Read an input file
    b) Calls the "filter" function of the filter model
    c) After SAMPLES_PER_OUTPUT_FILE samples, writes the filtered data to a file.
       Seperates the data in a few different files.
"""
# Standard imports
import argparse
import json
import logging
import pathlib
import pickle

# Third-party imports
import numpy as np
from sklearn import naive_bayes
from sklearn import preprocessing
import tqdm

# Our imports
import utils

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
    def __init__(self, model_config_path: utils.PathStr,
                 expected_json_keys):
        # Open and load the json with the configuration of the filter.
        with open(model_config_path) as fin:
            self._config = json.load(fin)

        # Validate the keys of the config file.
        if not self._config.keys() == expected_json_keys:
            raise ValueError(f"Received different keys than expected.\n"
                             f"Got:      {sorted(expected_json_keys)}\n"
                             f"Expected: {sorted(self._config)}")

    def filter(self, samples: np.ndarray
               ) -> np.ndarray:
        """ Returns a Numpy array with booleans, telling which samples to use.
        """

        # A pure abstract function is a function that can't be called by itself,
        # it needs to be derived by the derived classes
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
        - vocab_size: Most models need to know the vocab size; this one is not
            an exception. 
        - filter_threshold: Value at which the predictions become positive.
            for example, a prediction score of 0.321 and a threshold of 0.25
            gives a positive value.

    """
    # Expected keys of the json config file that has the specific
    # configuration of this filter
    expected_json_keys = {"model_pkl_path", "vocab_size", "filter_threshold"}

    # TODO(julesgm, im-ant): Should probably be moved to a separate file.
    def __init__(self, model_config_path: utils.PathStr):
        """ Loads the model.
        """
        # Call the constructor of the base class.
        super().__init__(model_config_path, self.expected_json_keys)

        # Load the NBC model that was trained with the trainer.
        with open(self._config["model_pkl_path"], "rb") as fin:
            # We specify the class of the model with an inline annotation
            self._model: naive_bayes.MultinomialNB = pickle.load(fin)

    def filter(self, samples: np.ndarray
               ) -> np.ndarray:
        """Filter the samples.
        Arguments:
            samples: 
                The samples from which we create a mask.
                We are assuming that they are of size BatchSize x SeqLen
                They Are indices.
        
        Returns:
            A numpy array with the boolean filter mask.
        """

        # Convert the int indices to one hot representation.
        oh_samples = utils.to_categorical(samples, 
            num_classes=self._config["num_classes"])
        
        # Add the one hot representations over the length of the sentence
        # to get a bag of word vector of the sentence.
        bow_samples = sum(oh_samples, 1)

        # Run the prediction.
        prediction_scores = self._model.predict(bow_samples)
        assert prediction_scores.shape == samples.shape[:2], (
            prediction_scores.shape, samples.shape[:2])
        
        # Threshold all the values to get the bolean mask.
        # TODO(julesgm): This part is error prone. Test more when live.
        return prediction_scores > self._config["filter_threshold"]

# Map mapping the names of the different filter types to their class.
# This allows us to receive their name by command-line argument, 
# and then construct the correct class.
# Multiple names can point to the same class.
FILTER_MAP = dict(naive_bayes_classifier=NBCFilter,
                  nbc=NBCFilter,     
                  # hand_written_rules=HandWrittenRulesFilter,
                  # etc.
                  )

def main(args: argparse.Namespace):
    # Log the command-line arguments in a pretty way.
    utils.log_args(args, logging.DEBUG)

    ###########################################################################
    # 1. Load filter
    ###########################################################################
    logging.info("Loading the filter.")
    # Construct the correct filter class for the argument we received in the
    # command line.
    filter_ = FILTER_MAP[args.filter_type](args.json_config_path)

    ###########################################################################
    # 2. Filter data & save it
    ###########################################################################
    # The format of the output data may change.
    # We are expecting small arrays of ints in numpy's npz format currently.
    
    positive_samples = []
    active_output_sample_count = 0
    output_file_index = 0
    
    # TODO(julesgm, im-ant): Do this in parallel. 
    # The stack thing is not ideal in parallel; it should be easy to come up with 
    # something else though.
    
    # Tqdm is an extremely popular progress bar library.
    # We iterate over all the files of Numpy's `.npz` format.
    for file_ in tqdm.tqdm(pathlib.Path(args.input_data_path).glob(f"*.npz")):
        logging.debug(f"Loading file \"{file_}\"")
        # Load the numpy array with the indices.
        samples = np.load(file_, dtype=int)
        logging.debug(f"Loaded file \"{file_}\".")
        
        logging.debug(f"Filtering the samples from a file.")
        # Get the mask from the filter object.
        mask = filter_.filter(samples)
        # Only select those where the mask was positive.
        new_output_samples = samples[mask]
        
        # Add them to our positive samples.
        positive_samples.append(new_output_samples)
        active_output_sample_count += len(new_output_samples)

        # We save the samples in a different file every SAMPLES_PER_OUTPUT_FILE
        # sample.
        if active_output_sample_count >= SAMPLES_PER_OUTPUT_FILE:
            concatenated_samples = np.concatenate(positive_samples)
            output_samples = concatenated_samples[:SAMPLES_PER_OUTPUT_FILE]

            # Awkward addition of the file number suffix. My searches suggest it is
            # the real way to do it. 
            logging.debug(f"Saving {output_file_index}.")
            with_suffix = pathlib.Path(str(args.output_data_path_prefix) 
                                       + f"_{output_file_index}")
            np.save(with_suffix, output_samples)
            logging.debug(f"Saved {output_file_index}.")
            output_file_index += 1

            # Adjust the count to fit the number of samples left in the `positive_samples`
            # queue.
            # TODO(julesgm): This part is error prone. Test more when live.
            if active_output_sample_count > SAMPLES_PER_OUTPUT_FILE:
                positive_samples = [concatenated_samples[SAMPLES_PER_OUTPUT_FILE:]]
                active_output_sample_count = len(positive_samples[0])
            else:
                positive_samples = []
                active_output_sample_count = 0

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