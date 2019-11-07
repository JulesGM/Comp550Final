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

import fire
import numpy as np
from sklearn import naive_bayes

import utils


class FilterAbstractTrainer:
    """Base class of all Filter Trainer instances.
    Requires that the filters can be trained and that their models
    can be saved. Also requires being able to accept a path in the 
    init to a json configuration file.
    """
    def __init__(self, config_path: utils.PathStr):
        with open(config_path) as fin:
            self._config = json.load(fin)

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
    def __init__(self, config_path: utils.PathStr):
        super().__init__(config_path)
        self._model = naive_bayes.MultinomialNB(
            alpha=self._config["hyperparams"].get("alpha", 1.0),
            fit_prior=self._config["hyperparams"].get("fit_prior", True),
            class_prior=self._config["hyperparams"].get("class_prior", None))


    def train(self, data_from_labeled_set: np.ndarray, 
              data_from_unlabeled_set: np.ndarray):
        utils.check_type(data_from_labeled_set, np.ndarray)
        utils.check_type(data_from_unlabeled_set, np.ndarray)

        # Make the dataset smaller. In the dumbest way possible.
        np.random.shuffle(data_from_unlabeled_set)
        data_from_unlabeled_set = data_from_unlabeled_set[:len(data_from_labeled_set)]

        y = np.concatenate([np.ones(dtype=int, shape=[len(data_from_labeled_set)]),
                            np.zeros(dtype=int, shape=[len(data_from_unlabeled_set)])])
        x = np.concatenate([data_from_labeled_set, data_from_unlabeled_set])

        # To One Hot
        x_oh = utils.to_categorical(x, num_classes=self._config["vocab_size"])
        
        # To BOW
        x_bow = np.sum(x_oh, axis=1)
    
        self._model.fit(x_bow, y)

    def save(self, path: utils.PathStr):
        logging.info("Saving Model.")
        with open(path) as fout:
            pickle.dump(self._model, fout)
        logging.info("Done saving Model.")

def load_unlabeled_data(path: utils.PathStr) -> np.ndarray:
    """Prototypical version of the unlabeled dataset loader.
    TODO(julesgm): Switch to not using npz
    """
    return np.load(path)

def load_flattened_labeled_data(path: utils.PathStr) -> np.ndarray:
    """Prototypical version of the flattened labeled dataset loader.
    TODO(julesgm): Switch to not using npz
    """
    return np.load(path)

MODEL_TYPE_MAP = dict(naive_bayes=NaiveBayesClassifierFilterTrainer,
                      naive_bayes_classifier=NaiveBayesClassifierFilterTrainer,
                      nbc=NaiveBayesClassifierFilterTrainer,
                      # ... Some more. Multiple names per entry are encouraged.
                      )

def main(flattened_labeled_data_path: utils.PathStr, input_data_path: utils.PathStr, 
         model_config_path: utils.PathStr, model_type: str,
         trainer_save_path: utils.PathStr, unlabeled_dataset_path: utils.PathStr, 
         verbosity: int = int(logging.DEBUG)):
    """
    Randomly loads an equal amount of unlabeled data to the size of the flattened
    labeled dataset, then trains the model to be used for smart filtering,
    then saves it.

    Which model specifically is to be used is specified by the `model_type` 
    argument. The choices are the keys of the `MODEL_TYPE_MAP` dict. Multiple
    keys refer to the same class to make it easier to use.

    Arguments:
        flattened_labeled_data_path: 
            Path of the data to be used to train the filter, after having 
            been transformed to the format that is compatible with the filter.
        input_data_path: 
            Path to the input data.
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
    utils.check_type(verbosity, int)    
    utils.check_type(model_type, str)
    utils.check_type(input_data_path, [str, pathlib.Path])
    utils.check_type(model_config_path, [str, pathlib.Path])
    utils.check_type(trainer_save_path, [str, pathlib.Path])
    utils.check_type(flattened_labeled_data_path, [str, pathlib.Path])
    utils.check_type(unlabeled_dataset_path, [str, pathlib.Path])

    # Argument Type Coercions
    input_path = pathlib.Path(input_data_path)
    model_config_path = pathlib.Path(model_config_path)
    trainer_save_path = pathlib.Path(trainer_save_path)
    flattened_labeled_data_path = pathlib.Path(flattened_labeled_data_path)
    unlabeled_dataset_path = pathlib.Path(unlabeled_dataset_path)

    # Argument Enum Checks
    model_type = model_type.lower()
    if model_type not in MODEL_TYPE_MAP.keys():
        raise ValueError(f"Invalid value for model_type. Got \"{model_type}\","
                         f" expected one of {set(MODEL_TYPE_MAP)}.")

    # Logger Setup
    logging.basicConfig(format='%(message)s')
    logger = logging.getLogger()
    logger.setLevel(verbosity)

    # Load Data
    data_from_labeled_set = load_flattened_labeled_data(flattened_labeled_data_path)
    data_from_unlabeled_set = load_unlabeled_data(unlabeled_dataset_path)

    # Trainer Action
    trainer = MODEL_TYPE_MAP[model_type](model_config_path)
    trainer.train(data_from_labeled_set, data_from_unlabeled_set)
    trainer.save(trainer_save_path)


if __name__ == "__main__":
    fire.Fire(main)
