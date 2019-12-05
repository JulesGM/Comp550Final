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
import glob
import json
import logging
import pathlib
import pickle

# Third-party imports
import numpy as np
from sklearn import naive_bayes
from sklearn import preprocessing
import tensorflow as tf
import tqdm

# Our imports
import tf_example_utils
import utils

SAMPLES_PER_OUTPUT_FILE = 1000
SAMPLE_LEN = 128


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

    def filter(self, samples: tf.Tensor) -> np.ndarray:
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
    expected_json_keys = {"vocab_size", "filter_threshold"}
    expected_merge_modes = {"either", "both"}

    # TODO(julesgm, im-ant): Should probably be moved to a separate file.
    def __init__(self, model_config_path: utils.PathStr, 
                 model_ckpt_path: utils.PathStr):
        """ Loads the model.
        Arguments:
            model_config_path: Where the config.json file is saved.
            model_ckpt_path: Pickle save of the model
        """
        # Call the constructor of the base class.
        super().__init__(model_config_path, self.expected_json_keys)

        # Load the NBC model that was trained with the trainer.
        with open(model_ckpt_path, "rb") as fin:
            # We specify the class of the model with an inline annotation
            self._model: naive_bayes.MultinomialNB = pickle.load(fin)

    def filter(self, samples: tf.Tensor) -> np.ndarray:
        """Filter the samples.
        Arguments:
            samples: 
                The samples from which we create a mask.
                We are assuming that they are of size BatchSize x SeqLen
                They Are indices.
        
        Returns:
            A numpy array with the boolean filter mask.
        """
    
        maxlen = max(map(len, samples))
        one_hot = tf.one_hot([tf.pad(sample, [[0, maxlen - len(sample)]]) 
                              for sample in samples], 
            self._config["vocab_size"])
    
        # Add the one hot representations over the length of the sentence
        # to get a bag of word vector of the sentence.
        bow_samples = tf.math.reduce_sum(one_hot, axis=1)
        prediction_scores = self._model.predict(bow_samples.numpy())

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
    filter_ = FILTER_MAP[args.filter_type](args.json_config_path,
                                           args.model_ckpt_path)

    ###########################################################################
    # 2. Filter data & save it
    ###########################################################################
    # The format of the output data may change.
    # We are expecting small arrays of ints in numpy's npz format currently.
    
    # TODO(julesgm, im-ant): Do this in parallel. 
    # The stack thing is not ideal in parallel; it should be easy to come up 
    # with something else though.
    
    reader = tf_example_utils.read_from_tf_example(
        glob.glob(str(args.input_data_path)), sample_len=SAMPLE_LEN, 
        shuffle_buffer_size=args.shuffle_buffer_size,
        num_map_threads=args.num_map_threads, 
        num_epochs=1)

    with tf_example_utils.WriteAsTfExample(output_files=[args.output_data_path],
                                           vocab_path=args.vocab_path, 
                                           max_num_tokens=SAMPLE_LEN) as writer:

        for i, batch in enumerate(reader.batch(args.batch_size)):        
            # Get the mask from the filter object.
            print(f"Batch {i}")

            sent_as = [batch["input_ids"][i][batch["segment_ids"][i] == 0][1: -1]
                       for i in range(len(batch["input_ids"]))]
            sent_bs = [batch["input_ids"][i][batch["segment_ids"][i] == 1]
                       for i in range(len(batch["input_ids"]))]
            sents = sent_as + sent_bs
            mask = filter_.filter(sents)
            if args.merge_mode == "both":
                mask = mask[::2] & mask[1::2]
            elif args.merge_mode == "either":
                mask = mask[::2] | mask[1::2]
            # Only select those where the mask was positive.
            
            new_output_samples = {k: v[tf.constant(mask, dtype=tf.bool)] 
                                  for k, v in batch.items()}
            
            print(len(new_output_samples["input_ids"]))
            print(len(new_output_samples["input_ids"]) / len(batch["input_ids"]))
            # Add them to our positive samples.
            writer.from_feature_batch(new_output_samples)

    logging.info("Done.")
                


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter_type", choices=list(FILTER_MAP.keys()), 
                        help="Which type of filter we are using.")
    parser.add_argument("--json_config_path", type=pathlib.Path,
                        help="Path of the model's configuration file.")
    parser.add_argument("--input_data_path", type=pathlib.Path,
                        help="Path to the data.")
    parser.add_argument("--output_data_path", type=pathlib.Path,
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
    parser.add_argument("--model_ckpt_path", type=pathlib.Path)
    parser.add_argument("--shuffle_buffer_size", type=int, 
                        help=("shuffle_buffer_size for tf.data.Dataset of "
                              "the main data loader."))
    parser.add_argument("--num_map_threads", type=int, 
                        help="Number of threads to use to de-serialize the "
                             "dataset.")
    parser.add_argument("--batch_size", type=int, 
                        help="Size of the batches.")
    parser.add_argument("--vocab_path", type=pathlib.Path,
                        help="Path of BERT's the vocabulary file.")
    parser.add_argument("--merge_mode", "-mm", help="How to deal with a "
                        "sentence when only one of the two sentences "
                        "are positive.", choices={"both", "either"},
                        default="either")
    args = parser.parse_args()

    # Logging
    FORMAT = '%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger()
    logger.setLevel(args.verbosity)
    
    main(args)