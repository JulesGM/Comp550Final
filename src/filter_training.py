"""Trains the models to be used as smart filters. Prototype!
The main way this is a prototype is that the format of the data will be different,
and that the way we are subsampling the unlabeled dataset will be different.
Right now, it is using the numpy .npz format, but that's a super unefficient format,
because it is a dense format (which would make most of the space being used be 
for padding).
We'll roll with it for the moment.
"""
import json
import logging
import pathlib
import pickle
from typing import Set

# The following import is a conditional import.
try:
    import colored_traceback.auto
except ImportError:
    pass
import fire
import numpy as np
from sklearn import naive_bayes
import tensorflow as tf
import tqdm

import tf_example_utils
import utils

SAMPLE_LEN = 128

class FilterAbstractTrainer:
    """Base class of all Filter Trainer instances.
    Requires that the filters can be trained and that their models
    can be saved. Also requires being able to accept a path in the 
    init to a json configuration file.
    """
    def __init__(self, config_path: utils.PathStr, 
                 expected_json_keys: Set[str] 
                 # We require a set to check equality.
                 ):
        # Open the file and load the configuration file.
        with open(config_path) as fin:
            self._config = json.load(fin)

        # Verify that the keys of the json file match the expected ones
        # exactly.
        if not self._config.keys() == expected_json_keys:
                raise ValueError(f"Received different keys than expected.\n"
                                f"Got:      {sorted(expected_json_keys)}\n"
                                f"Expected: {sorted(self._config)}")

    def train(self, data_from_labeled_set, data_from_unlabeled_set):
        """Train the model of the smart filter.
        """
        raise NotImplementedError("Pure Abstract Method")

    def save(self, path: utils.PathStr):
        """Save the model of the smart filter.
        This is so it can then be loaded by the filtering module.
        """
        raise NotImplementedError("Pure Abstract Method")


class NaiveBayesClassifierFilterTrainer(FilterAbstractTrainer):
    """Smart filter using a Naive Bayes Classifier.
    Saves itself with the pickle module. Prototypical implementation.
    """
    expected_json_keys = {"hyperparams", "vocab_size", "data_prep_batch_size"}

    def __init__(self, config_path: utils.PathStr):
        super().__init__(config_path, self.expected_json_keys)
        
        # Initialize the model.
        # Here we use `dict.get` to have default values to the hyperparameters.
        self._model = naive_bayes.MultinomialNB(
            alpha=self._config["hyperparams"].get("alpha", 1.0),
            fit_prior=self._config["hyperparams"].get("fit_prior", True),
            class_prior=self._config["hyperparams"].get("class_prior", None))


    def train(self, data_from_labeled_set: tf.data.Dataset, 
              data_from_unlabeled_set: tf.data.Dataset):
        utils.check_type(data_from_labeled_set, [tf.data.Dataset])
        utils.check_type(data_from_unlabeled_set, [tf.data.Dataset])

        data_from_labeled_set = tf.stack([sample["input_ids"] for sample in 
                                         data_from_labeled_set])
        data_from_unlabeled_set = tf.stack([sample["input_ids"] for sample in 
                                           data_from_unlabeled_set])
        
        # TODO(julesgm, im-ant): This will not work when the dataset is huge.
        tf.random.shuffle(data_from_unlabeled_set)
        data_from_unlabeled_set = data_from_unlabeled_set[
            :len(data_from_labeled_set)]
        
        # Build the labels for our dataset.
        y = np.concatenate([np.ones(dtype=int, 
                                    shape=[len(data_from_labeled_set)]),
                            np.zeros(dtype=int, 
                                     shape=[len(data_from_unlabeled_set)])])

        # Concatenate the `positive` and the `negative` examples.
        x = tf.concat([data_from_labeled_set, data_from_unlabeled_set], axis=0)

        pre_concat = []
        batch_size = self._config["data_prep_batch_size"]
        upper_bound = x.shape[0] // batch_size + 1
        logging.info("Creating Bag of Word Features")
        for i in tqdm.tqdm(range(upper_bound)):
            batch = x[i * batch_size:(i + 1) * batch_size]

            # Convert the id sequences to one hot representation.
            x_oh = tf.one_hot(batch, self._config["vocab_size"])
            
            # Add the one hot representations to get a per-sentence bag of word.
            pre_concat.append(tf.math.reduce_sum(x_oh, axis=1))
            
            del batch
            del x_oh 
        del x
        x_bow = tf.concat(pre_concat, axis=0)
        del pre_concat

        logging.info("Fitting Model")
        self._model.fit(x_bow.numpy(), y)

    def save(self, path: utils.PathStr) -> None:
        """Save the model once we have trained it.
        """
        logging.info("Saving Model.")
        # Open the file..
        with open(str(path), "wb") as fout:
            # Dump the object.
            pickle.dump(self._model, fout)
        logging.info("Done saving Model.")


def load_data(path: utils.PathStr, num_map_threads: int = 4,
              num_epochs: int = 1, shuffle_buffer_size: int = 1,
              force: bool = False,
              ) -> tf.data.Dataset:
    """Prototypical version of the flattened labeled dataset loader.
    
    Arguments:  
        path: Paths of the tf.Example file.
        sample_len: Length in number of tokens of the sample.
        num_epochs: Number of times to loop over the data.
        num_map_threads: Number fo threads to use for the deserialization
                        of the tf.Example bytes to Python (Tensorflow) objects.
        shuffle_buffer_size: the data is loaded this number of samples at the 
                             time, and these "shuffle batches" have their sample 
                             shuffled.
    Returns:
        A tf.data.Dataset object that returns the samples one at the time.
    """
    # This is kind of dumb, but the whole dataset is very small,
    # so there is no problem
    return tf_example_utils.readFromTfExample([path], 
        sample_len=SAMPLE_LEN, shuffle_buffer_size=shuffle_buffer_size,
        num_map_threads=num_map_threads, num_epochs=num_epochs)

    
# Like in filter_inference, this is a map between the filter names
# and their classes. This allows us to receive the name as an argument,
# and construct the correct class.
MODEL_TYPE_MAP = dict(naive_bayes=NaiveBayesClassifierFilterTrainer,
                      naive_bayes_classifier=NaiveBayesClassifierFilterTrainer,
                      nbc=NaiveBayesClassifierFilterTrainer,
                      # ... Some more. Multiple names per entry are encouraged.
                      )

def main(flattened_labeled_data_path: utils.PathStr, 
         model_config_path: utils.PathStr, model_type: str,
         trainer_save_path: utils.PathStr, unlabeled_dataset_path: utils.PathStr, 
         verbosity: int = int(logging.DEBUG), force: bool = False):
    """
    Randomly loads an equal amount of unlabeled data to the size of the 
    flattened labeled dataset, then trains the model to be used for smart 
    filtering, then saves it.

    Which model specifically is to be used is specified by the `model_type` 
    argument. The choices are the keys of the `MODEL_TYPE_MAP` dict. Multiple
    keys refer to the same class to make it easier to use.

    Some of the arguments could probably be moved to the configuration file.

    Arguments:
        flattened_labeled_data_path: 
            Path of the data to be used to train the filter, after having 
            been transformed to the format that is compatible with the filter.
        model_config_path:
            Path to the json config file.
        model_type:
            Type of model to train.
        trainer_save_path:
            Path of where we should save the model.
        unlabeled_dataset_path:
            Path to the unlabeled dataset.
        verbosity:
            Verbosity levels in python: 
                NOTSET = 0
                DEBUG = 10
                INFO = 20
                WARNING = 30 
                WARN = WARNING
                ERROR = 40 
                CRITICAL = 50 
                FATAL = CRITICAL         
    Returns:
        Void
    """
    # Argument Type Checks
    utils.check_type(verbosity, [int])    
    utils.check_type(model_type, [str])
    utils.check_type(model_config_path, [str, pathlib.Path])
    utils.check_type(trainer_save_path, [str, pathlib.Path])
    utils.check_type(flattened_labeled_data_path, [str, pathlib.Path])
    utils.check_type(unlabeled_dataset_path, [str, pathlib.Path])

    # Argument Type Coercions
    model_config_path = pathlib.Path(model_config_path)
    trainer_save_path = pathlib.Path(trainer_save_path)
    flattened_labeled_data_path = pathlib.Path(flattened_labeled_data_path)
    unlabeled_dataset_path = pathlib.Path(unlabeled_dataset_path)

    if force or not trainer_save_path.exists():
        # Check that the model type is one of the ones we can handle.
        model_type = model_type.lower()
        if model_type not in MODEL_TYPE_MAP.keys():
            raise ValueError(f"Invalid value for model_type. Got "
                             f"\"{model_type}\", expected one of "
                             f"{set(MODEL_TYPE_MAP)}.")

        # Logger Setup
        logging.basicConfig(format='%(message)s')
        logger = logging.getLogger()
        logger.setLevel(verbosity)

        # Load Data
        data_from_labeled_set = load_data(flattened_labeled_data_path)
        data_from_unlabeled_set = load_data(unlabeled_dataset_path)

        # Trainer Action
        trainer = MODEL_TYPE_MAP[model_type](model_config_path)
        trainer.train(data_from_labeled_set, data_from_unlabeled_set)
        trainer.save(trainer_save_path)
        logging.info("Done.")

if __name__ == "__main__":
    fire.Fire(main)
