"""Trains the models to be used as smart filters. Prototype!
The main way this is a prototype is that the format of the data will be different,
and that the way we are subsampling the unlabeled dataset will be different.
Right now, it is using the numpy .npz format, but that's a super unefficient format,
because it is a dense format (which would make most of the space being used be 
for padding).
We'll roll with it for the moment.
"""
import glob
import json
import logging
import itertools
import pathlib
import pickle
from typing import Set
import random

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
from typing import Any, Dict, List

import tf_example_utils
import utils

NUMBER_TO_SAMPLE = 16712  # SocialIQA ... this is a bit dirty


class FilterAbstractTrainer:
    """Base class of all Filter Trainer instances.
    Requires that the filters can be trained and that their models
    can be saved. Also requires being able to accept a path in the
    init to a json configuration file.
    """

    def __init__(self, config_path: utils.PathStr,
                 expected_json_keys: Set[str],
                 vocab_path: utils.PathStr
                 # We require a set to check equality.
                 ):
        # Open the file and load the configuration file.
        with open(vocab_path) as fin:
            self._idx_to_w = [w.strip() for w in fin]

        with open(config_path) as fin:
            self._config = json.load(fin)

        # Verify that the keys of the json file match the expected ones
        # exactly.
        if not self._config.keys() == expected_json_keys:
            raise ValueError(f"Received different keys than expected.\n"
                             f"Got:      {sorted(expected_json_keys)}\n"
                             f"Expected: {sorted(self._config)}")

    def train(self, data_from_labeled_set: tf.data.Dataset,
              data_from_unlabeled_set: List[tf.Tensor]) -> None:
        """Train the model of the smart filter.
        """
        raise NotImplementedError("Pure Abstract Method")

    def save(self, path: utils.PathStr) -> None:
        """Save the model of the smart filter.
        This is so it can then be loaded by the filtering module.
        """
        raise NotImplementedError("Pure Abstract Method")

    @property
    def batch_size(self) -> int:
        return self._config["batch_size"]

    @batch_size.setter
    def batch_size(self, val):
        raise RuntimeError("Can't assign to the batch size this way.")


class TransformerEncoderFilterTrainer(FilterAbstractTrainer):
    expected_json_keys = {"vocab_size", "dimension",
                          "dropout", "max_len", "fc_multiplier", "num_layers",
                          "batch_size"}

    def __init__(self, config_path: utils.PathStr, vocab_path: utils.PathStr):
        super().__init__(config_path, self.expected_json_keys,
                         vocab_path=vocab_path)

        dimension: int = self._config["dimension"]
        vocab_size: int = self._config["vocab_size"]
        dropout: float = self._config["dropout"]
        max_len: int = self._config["max_len"]
        num_layers: int = self._config["num_layers"]
        fc_multiplier: int = self._config["fc_multiplier"]

        token_ids = tf.keras.Input(shape=(None,), dtype='int32')
        # position_ids = tf.keras.Input(shape=(None,), dtype='int32')

        # Embedding lookup.
        token_embedding = tf.keras.layers.Embedding(vocab_size, dimension)
        positon_embedding = tf.keras.layers.Embedding(max_len, dimension)

        # Query embeddings of shape [batch_size, Tq, dimension].
        x = token_embedding(token_ids) + positon_embedding(
                tf.range(max_len))
        for i in range(num_layers):
            x = tf.keras.layers.Dropout(dropout)(x)
            fc_q = tf.keras.layers.Dropout(dropout)(
                    tf.keras.layers.Conv1D(dimension, 1)(x))
            fc_k = tf.keras.layers.Dropout(dropout)(
                    tf.keras.layers.Conv1D(dimension, 1)(x))
            fc_v = tf.keras.layers.Dropout(dropout)(
                    tf.keras.layers.Conv1D(dimension, 1)(x))
            att = tf.keras.layers.Dropout(dropout)(
                    tf.keras.layers.Attention()([fc_q, fc_k, fc_v],
                                              [token_ids != 0, token_ids != 0]))
            fc_o = tf.keras.layers.Conv1D(dimension, 1)(att)
            x = tf.keras.layers.LayerNormalization()(fc_o + x)

            fc_in = tf.keras.layers.Conv1D(dimension * fc_multiplier, 1,
                                           activation="relu")(
                    tf.keras.layers.Dropout(dropout)(x))
            fc_out = tf.keras.layers.Conv1D(dimension, 1)(
                    tf.keras.layers.Dropout(dropout)(fc_in))
            x = tf.keras.layers.LayerNormalization()(fc_out + x)

        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        self._model = tf.keras.Model(inputs=token_ids,
                                     outputs=x)

    def train(self, data_from_labeled_set: tf.data.Dataset,
              data_from_unlabeled_set: List[tf.Tensor],):

        data_from_labeled_set = [x["input_ids"] for x in
                                 tqdm.tqdm(data_from_labeled_set)]
        data_from_unlabeled_set = [x["input_ids"] for x in
                                   tqdm.tqdm(data_from_unlabeled_set)]
        assert len(data_from_labeled_set) == len(data_from_unlabeled_set)

        print("\n#########################################################")
        print("#########################################################")
        print("#########################################################\n")

        print(f"len(data_from_labeled_set): {len(data_from_labeled_set)}")

        print("\n#########################################################")
        print("#########################################################")
        print("#########################################################\n")

        print(f"len(data_from_unlabeled_set): {len(data_from_unlabeled_set)}")

        print("\n#########################################################")
        print("#########################################################")
        print("#########################################################\n")

        y = tf.concat(
            [tf.zeros(dtype=tf.int64, shape=[len(data_from_labeled_set), 1]),
             tf.ones(dtype=tf.int64, shape=[len(data_from_unlabeled_set), 1])],
                      axis=0)

        for _ in range(10):
            idx = random.choice(range(len(data_from_labeled_set)))
            print(idx)
            print(" ".join([self._idx_to_w[i]
                        for i in data_from_labeled_set[idx] if i != 0]))

        print("\n#########################################################")
        print("#########################################################")
        print("#########################################################\n")

        for _ in range(10):
            idx = random.choice(range(len(data_from_unlabeled_set)))
            print(idx)
            print(" ".join([self._idx_to_w[i]
                        for i in data_from_unlabeled_set[idx] if i != 0]))
        x_ids = tf.stack(data_from_labeled_set + data_from_unlabeled_set)
        # x_pos = (np.arange(128) for x in itertools.count())

        self._model.compile(optimizer="adam", loss="binary_crossentropy",
                            metrics=["accuracy"])
        self._model.fit(x=x_ids, y=y, batch_size=self.batch_size,
                        shuffle=True, verbose=True)

    def save(self, path: utils.PathStr) -> None:
        self._model.save(path)


class LSTMFilterTrainer(FilterAbstractTrainer):
    expected_json_keys = {"vocab_size", "dimension", "max_len",
                          "dropout", "num_layers", "batch_size"}

    def __init__(self, config_path: utils.PathStr, vocab_path: utils.PathStr):
        super().__init__(config_path, self.expected_json_keys, vocab_path)
        dimension: int = self._config["dimension"]
        vocab_size: int = self._config["vocab_size"]
        dropout: float = self._config["dropout"]
        max_len: int = self._config["max_len"]
        num_layers: int = self._config["num_layers"]

        token_ids = tf.keras.Input(shape=(None,), dtype='int32')

        # Embedding lookup.
        token_embedding = tf.keras.layers.Embedding(vocab_size, dimension,
                                                    mask_zero=True)

        # Query embeddings of shape [batch_size, Tq, dimension].
        x = token_embedding(token_ids)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                    dimension),
                    input_shape=(max_len, dimension))(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        self._model = tf.keras.Model(inputs=token_ids,
                                     outputs=x)

    def train(self, data_from_labeled_set: tf.data.Dataset,
              data_from_unlabeled_set: List[tf.Tensor],):

        data_from_labeled_set = [x["input_ids"] for x in
                                 tqdm.tqdm(data_from_labeled_set)]
        data_from_unlabeled_set = [x["input_ids"] for x in
                                   tqdm.tqdm(data_from_unlabeled_set)]
        assert len(data_from_labeled_set) == len(data_from_unlabeled_set)

        print("\n#########################################################")
        print("#########################################################")
        print("#########################################################\n")

        print(f"len(data_from_labeled_set): {len(data_from_labeled_set)}")

        print("\n#########################################################")
        print("#########################################################")
        print("#########################################################\n")

        print(f"len(data_from_unlabeled_set): {len(data_from_unlabeled_set)}")

        print("\n#########################################################")
        print("#########################################################")
        print("#########################################################\n")

        y = tf.concat(
            [tf.zeros(dtype=tf.int64, shape=[len(data_from_labeled_set), 1]),
             tf.ones(dtype=tf.int64, shape=[len(data_from_unlabeled_set), 1])],
                      axis=0)

        for _ in range(10):
            idx = random.choice(range(len(data_from_labeled_set)))
            print(idx)
            print(" ".join([self._idx_to_w[i]
                        for i in data_from_labeled_set[idx] if i != 0]))

        print("\n#########################################################")
        print("#########################################################")
        print("#########################################################\n")

        for _ in range(10):
            idx = random.choice(range(len(data_from_unlabeled_set)))
            print(idx)
            print(" ".join([self._idx_to_w[i]
                        for i in data_from_unlabeled_set[idx] if i != 0]))
        x_ids = tf.stack(data_from_labeled_set + data_from_unlabeled_set)

        self._model.compile(optimizer="adam", loss="binary_crossentropy",
                            metrics=["accuracy"])
        self._model.fit(x=x_ids, y=y, batch_size=self.batch_size,
                        shuffle=True, verbose=True)

    def save(self, path: utils.PathStr) -> None:
        self._model.save(path)


class NaiveBayesClassifierFilterTrainer(FilterAbstractTrainer):
    """Smart filter using a Naive Bayes Classifier.
    Saves itself with the pickle module. Prototypical implementation.
    """
    expected_json_keys = {"hyperparams", "vocab_size", "batch_size"}

    def __init__(self, config_path: utils.PathStr, vocab_path: utils.PathStr):
        super().__init__(config_path, self.expected_json_keys, vocab_path)

        # Initialize the model.
        # Here we use `dict.get` to have default values to the hyperparameters.

        alpha: float = utils.check_type_one_of(
                self._config["hyperparams"]["alpha"], {float})

        fit_prior: bool = utils.check_type_one_of(
                self._config["hyperparams"]["fit_prior"], {bool})

        self._model = naive_bayes.MultinomialNB(alpha=alpha,
                                                fit_prior=fit_prior)

    @staticmethod
    def _stack_per_sent(samples_a, samples_b):
        """Extract both of the sentences of a sample, and stack all of them.
        We need both sets of samples because we need to pad the the longuest
        sentence of both sets.

        There are two sentences in a sample. We want to train the filter
        as if they were independent samples. So, we extract the sentences from
        the samples by using the segment_ids. We added a third segment id
        for the padding in order to not get the padding when we filter
        with the segment_ids.
        """

        lengths = []
        packs = []

        for i, samples in enumerate([samples_a, samples_b]):
            # The weird [1:-1] is to remove the <cls> token and the <sep>
            # token from the first sentenceof a sample

            sents_0 = [sample["input_ids"][sample["segment_ids"] == 0][1:-1] for
                       sample in tqdm.tqdm(samples)]
            sents_1 = [sample["input_ids"][sample["segment_ids"] == 1] for
                       sample in tqdm.tqdm(samples)]
            if i == 1:
                for sample in itertools.islice(samples, 0, 100, 10):
                    print(sample["segment_ids"])

            # itertools.chain just .. chains the iteration over two iterables.
            # like, [x for x in itertools.chain(range(3), range(3))] would be
            # [0, 1, 2, 0, 1, 2]
            length = max(itertools.chain(map(len, sents_0), map(len, sents_1)))
            packs.append((sents_0, sents_1))
            lengths.append(length)

        maxlen = max(lengths)
        output = []

        for pack in tqdm.tqdm(packs):
            sents = [tf.pad(sent, [[0, maxlen - len(sent)]])
                     for sent in itertools.chain(*pack)]
            output.append(sents)

        return tf.stack(output[0]), tf.stack(output[1])

    def train(self, data_from_labeled_set: tf.data.Dataset,
              data_from_unlabeled_set: List[tf.Tensor]):
        logging.info("Split and Stack Sentences")
        data_from_labeled_set, data_from_unlabeled_set = self._stack_per_sent(
                data_from_labeled_set, data_from_unlabeled_set)

        assert len(data_from_labeled_set) == len(data_from_unlabeled_set)
        x = tf.concat([data_from_labeled_set, data_from_unlabeled_set], axis=0)

        pre_concat = []

        upper_bound = x.shape[0] // self.batch_size + 1
        logging.info("Creating Bag of Word Features")
        for i in tqdm.tqdm(range(upper_bound)):
            batch = x[i * self.batch_size:(i + 1) * self.batch_size]

            # Convert the id sequences to one hot representation.
            x_oh = tf.one_hot(batch, self._config["vocab_size"])

            # Add the one hot representations to get a per-sentence bag of word.
            pre_concat.append(tf.math.reduce_sum(x_oh, axis=1))

            del batch
            del x_oh
        del x
        x_bow = tf.concat(pre_concat, axis=0)
        del pre_concat

        # Build the labels for our dataset.
        y = np.concatenate([np.ones(dtype=int,
                                    shape=[len(data_from_labeled_set)]),
                            2 * np.ones(dtype=int,
                                        shape=[len(data_from_unlabeled_set)])])

        assert y.shape == (x_bow.shape[0],), y.shape

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


def load_data(paths: List[utils.PathStr], num_map_threads: int, sample_len: int
              ) -> tf.data.Dataset:
    """Prototypical version of the flattened labeled dataset loader.
    
    Arguments:  
        paths:
            paths of the tf.Example files.
        num_map_threads:
            Number fo threads to use for the deserialization of the tf.Example
            bytes to Python (Tensorflow) objects.
        sample_len:
            What is the maximum number of tokens.
    Returns:
        A tf.data.Dataset object that returns the samples one at the time.
    """
    # This is kind of dumb, but the whole dataset is very small,
    # so there is no problem
    if not paths:
        raise ValueError()
    parser_fn = tf_example_utils.build_filter_input_parser_fn(sample_len)
    return tf_example_utils.read_from_tf_example(paths, sample_len=sample_len,
                                                 shuffle_buffer_size=1,
                                                 num_map_threads=num_map_threads,
                                                 num_epochs=1,
                                                 parser_fn=parser_fn,
                                                 sharding_quantity=0,
                                                 sharding_idx=None,
                                                 )


# Like in filter_inference, this is a map between the filter names
# and their classes. This allows us to receive the name as an argument,
# and construct the correct class.
MODEL_TYPE_MAP = dict(naive_bayes=NaiveBayesClassifierFilterTrainer,
                      naive_bayes_classifier=NaiveBayesClassifierFilterTrainer,
                      nbc=NaiveBayesClassifierFilterTrainer,


                      trans=TransformerEncoderFilterTrainer,
                      lstm=LSTMFilterTrainer
                      )


def main(model_config_path: utils.PathStr,
         model_type: str, trainer_save_path: utils.PathStr,
         glob_pattern_unlabeled_data: utils.PathStr,
         glob_pattern_labeled_data: utils.PathStr, vocab_path: utils.PathStr,
         num_threads_reader: int, verbosity: int = int(logging.DEBUG),
         sample_len: int = 128):
    """
    Randomly loads an equal amount of unlabeled data to the size of the 
    flattened labeled dataset, then trains the model to be used for smart 
    filtering, then saves it.

    Which model specifically is to be used is specified by the `model_type` 
    argument. The choices are the keys of the `MODEL_TYPE_MAP` dict. Multiple
    keys refer to the same class to make it easier to use.

    Some of the arguments could probably be moved to the configuration file.

    Arguments:
        model_config_path:
            Path to the json config file.
        model_type:
            Type of machine learning model of the classifier to train.
        num_threads_reader:
            How many
        glob_pattern_labeled_data:
            Path of where we should save the model.
        glob_pattern_unlabeled_data:
            Path to the unlabeled dataset.
        trainer_save_path:
            Where to save the model.
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
        sample_len:
            maximum length of the sequences
    Returns:
        Void
    """
    # Argument Type Checks
    utils.check_type_one_of(verbosity, [int])
    utils.check_type_one_of(model_type, [str])
    utils.check_type_one_of(model_config_path, [str, pathlib.Path])
    utils.check_type_one_of(trainer_save_path, [str, pathlib.Path])
    utils.check_type_one_of(glob_pattern_labeled_data, [str, pathlib.Path])
    utils.check_type_one_of(glob_pattern_unlabeled_data, [str, pathlib.Path])

    # Argument Type Coercions
    model_config_path = pathlib.Path(model_config_path)
    trainer_save_path = pathlib.Path(trainer_save_path)
    glob_pattern_labeled_data = pathlib.Path(glob_pattern_labeled_data)
    glob_pattern_unlabeled_data = pathlib.Path(glob_pattern_unlabeled_data)

    # if force or not trainer_save_path.exists():
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
    logging.debug(str(glob_pattern_labeled_data))
    files_labeled = list(glob.glob(str(glob_pattern_labeled_data)))
    data_from_labeled_set = list(load_data(paths=files_labeled,
        num_map_threads=num_threads_reader, sample_len=sample_len,))

    logging.debug(str(glob_pattern_unlabeled_data))
    files_unlabeled = glob.glob(str(glob_pattern_unlabeled_data))

    data_from_unlabeled_set = list(tf_example_utils.tf_example_uniform_sampler(
                    paths=files_unlabeled, num_map_threads=num_threads_reader,
                    number_to_sample=len(data_from_labeled_set),
                    parser_fn=tf_example_utils.build_filter_input_parser_fn(sample_len)))

    # Trainer Action
    trainer = MODEL_TYPE_MAP[model_type](model_config_path, vocab_path)
    trainer.train(data_from_labeled_set, data_from_unlabeled_set)
    trainer.save(str(trainer_save_path))
    logging.info("Done.")


if __name__ == "__main__":
    fire.Fire(main)
