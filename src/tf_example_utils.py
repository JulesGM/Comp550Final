import dataclasses
import collections
import logging
import pathlib
import random
import time
from typing import Any, Callable, Dict, Iterable, List, Tuple, Type, Union

try:
    import colored_traceback.auto
except ImportError:
    pass
import numpy as np
import tensorflow as tf
import tqdm
from typing import TypeVar, Optional

import utils

CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"


def _truncate_seq_pair(tokens_a: list, tokens_b: list, max_num_tokens: int,
                       rng) -> None:
    """Copy pasted from https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L418

    Truncates a pair of sequences to a maximum sequence length.
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def _create_int_feature(values, feature_len):
    feature_list = list(values)
    utils.check_equal(len(feature_list), feature_len)
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=feature_list)
                               )
    return feature


def _create_float_feature(values, feature_len):
    feature_list = list(values)
    utils.check_equal(len(feature_list), feature_len)

    feature = tf.train.Feature(float_list=tf.train.FloatList(
            value=list(feature_list)))
    return feature


class TfRecordWriter:
    def __init__(self, output_files: List[utils.PathStr],
                 vocab_path: utils.PathStr, max_num_tokens: int):
        vocab_path = pathlib.Path(vocab_path)
        output_files = [pathlib.Path(output_path)
                        for output_path in output_files]

        # Open the vocab file
        with open(vocab_path) as fin:
            # Vocabulary mapping of the token text to their BERT id
            text_to_id = {text.strip(): id_ for id_, text in enumerate(fin)}

        self._cls_token_id = text_to_id[CLS_TOKEN]
        self._sep_token_id = text_to_id[SEP_TOKEN]
        self._max_num_tokens = max_num_tokens
        self._writers = [tf.io.TFRecordWriter(str(output_path))
                         for output_path in output_files]
        self._writer_index = 0

    def __enter__(self):
        return self

    def __exit__(self, type_, value, tb):
        self.close()

    def _write_one(self, feature_dict):
        # This could be made parallel with mutexes ...
        # Threads would would because writer.write is io bound
        # and (we hope) it releases the GIL
        tf_example = tf.train.Example(features=tf.train.Features(
                feature=feature_dict))
        self._writers[self._writer_index].write(tf_example.SerializeToString())
        self._writer_index = (self._writer_index + 1) % len(self._writers)

    def close(self):
        for writer in self._writers:
            writer.close()


@dataclasses.dataclass
class MLMInstance:
    __slots__ = ["index", "label"]
    index: int
    label: str


def create_masked_lm_predictions(ids: List[int], masked_lm_prob: float,
                                 max_predictions_per_seq, idx_to_words, rng,
                                 word_to_idx: Dict[str, int],
                                 do_whole_word_mask: bool = True,
                                 ):
    """Mostly copy pasted from BERT.
    Creates the predictions for the masked LM objective.
    """
    tokens = [idx_to_words[id_] for id_ in ids]
    cand_indexes = []

    for i, (token, id_) in enumerate(zip(tokens, ids)):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (do_whole_word_mask and len(cand_indexes) >= 1 and
                token.startswith("##")):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = idx_to_words[
                        rng.randint(0, len(idx_to_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MLMInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return ([word_to_idx[w] for w in output_tokens], masked_lm_positions,
            [word_to_idx[w] for w in masked_lm_labels])


class BERTExampleWriter(TfRecordWriter):
    def __init__(self, output_files: List[utils.PathStr],
                 vocab_path: utils.PathStr, max_num_tokens: int,):
        """Multiple output_files for if we want to save in multiple files.
        """
        super().__init__(output_files=output_files, vocab_path=vocab_path,
                         max_num_tokens=max_num_tokens,)

    T = TypeVar("T")
    @staticmethod
    def _pad(feature: List[T], length: int, padding_value: T) -> List[T]:
        while len(feature) < length:
            feature.append(padding_value)
        return feature[:length]


    def from_feature_batch(self, batch, idx_to_words, word_to_idx,
                           masked_lm_prob: float, max_predictions_per_seq=0):
        """
        Weird dict unbatching.
        """
        if not (len(batch["input_ids"]) == len(batch["input_mask"])
                == len(batch["segment_ids"])):
            raise RuntimeError("One feature had a weird length.")

        for input_ids, input_mask, segment_ids, next_sentence_label in zip(
                batch["input_ids"], batch["input_mask"], batch["segment_ids"],
                batch["next_sentence_labels"]):

            input_ids, masked_lm_positions, masked_lm_ids = (
                create_masked_lm_predictions(input_ids,
                    masked_lm_prob=masked_lm_prob,
                    max_predictions_per_seq=max_predictions_per_seq,
                    idx_to_words=idx_to_words, rng=random,
                    word_to_idx=word_to_idx, do_whole_word_mask=False))

            masked_lm_weights = [1.0] * len(masked_lm_ids)

            features = collections.OrderedDict()
            features["input_ids"] = _create_int_feature(input_ids,
                                                        self._max_num_tokens)
            features["input_mask"] = _create_int_feature(input_mask,
                                                         self._max_num_tokens)
            features["segment_ids"] = _create_int_feature(segment_ids,
                                                          self._max_num_tokens)
            features["masked_lm_positions"] = _create_int_feature(
                    self._pad(masked_lm_positions, max_predictions_per_seq, 0),
                    max_predictions_per_seq)
            features["masked_lm_ids"] = _create_int_feature(
                    self._pad(masked_lm_ids, max_predictions_per_seq, 0),
                    max_predictions_per_seq)
            features["masked_lm_weights"] = _create_float_feature(
                    self._pad(masked_lm_weights, max_predictions_per_seq, 0.0),
                    max_predictions_per_seq)
            features["next_sentence_labels"] = _create_int_feature(
                    [next_sentence_label], 1)


            self._write_one(features)


        # In one beautiful // terrible line. Works because dicts are
        # ordered now:
        # [self._write_one({k: v for k, v in zip(batch, vs)})
        #  for vs in zip(*batch.values())]


class WriteAsTfExample(TfRecordWriter):
    """Converts the token to tf_examples and writes them to drive.
    
    Can be used in a `with` statement:
    ```
    with WriteAsTfExample(output_paths, "vocab.txt", 128) as writer:
        for sent_a, sent_b, b_is_random in zip(a_sents, b_sents, b_rands):
            writer.add_sample(sent_a, sent_b, b_is_random)
    ```

    Please call `self.close()` if *not* used in a with statement.


    Strongly inspired by the code at :
    https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L131
    and at :
    https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L304
    """

    def __init__(self, output_files: List[utils.PathStr],
                 vocab_path: utils.PathStr, max_num_tokens: int):
        """Multiple output_files for if we want to save in multiple files.
        """
        super().__init__(output_files=output_files, vocab_path=vocab_path,
                         max_num_tokens=max_num_tokens)

    def add_sample(self, bpe_ids_a: List[int], bpe_ids_b: List[int],
                   b_is_random: bool) -> None:
        """Convert to tf.Example, then write to a file.
        """

        # We copy the lists because truncate_seq_pairs modifies them
        bpe_ids_a = list(bpe_ids_a)
        bpe_ids_b = list(bpe_ids_b)

        # -3 Because of the CLS token and of the two SEP tokens
        _truncate_seq_pair(bpe_ids_a, bpe_ids_b, self._max_num_tokens - 3,
                           random)
        bpe_ids = []
        segment_ids = []
        bpe_ids.append(self._cls_token_id)
        segment_ids.append(0)
        for token in bpe_ids_a:
            bpe_ids.append(token)
            segment_ids.append(0)

        bpe_ids.append(self._sep_token_id)
        segment_ids.append(0)

        for token in bpe_ids_b:
            bpe_ids.append(token)
            segment_ids.append(1)
        bpe_ids.append(self._sep_token_id)
        segment_ids.append(1)

        input_mask = len(bpe_ids) * [1]

        while len(bpe_ids) < self._max_num_tokens:
            bpe_ids.append(0)
            segment_ids.append(2)
            input_mask.append(0)

        next_sentence_label = 1 if b_is_random else 0

        features = collections.OrderedDict()
        features["input_ids"] = _create_int_feature(bpe_ids,
                                                    self._max_num_tokens)
        features["input_mask"] = _create_int_feature(input_mask,
                                                     self._max_num_tokens)
        features["segment_ids"] = _create_int_feature(segment_ids,
                                                      self._max_num_tokens)
        features["next_sentence_labels"] = _create_int_feature(
                [next_sentence_label], 1)

        self._write_one(features)


def build_filter_input_parser_fn(sample_len: int):
    feature_description = {
            "input_ids": tf.io.FixedLenFeature([sample_len], tf.int64),
            "input_mask": tf.io.FixedLenFeature([sample_len], tf.int64),
            "segment_ids": tf.io.FixedLenFeature([sample_len], tf.int64),
            "next_sentence_labels": tf.io.FixedLenFeature([1], tf.int64)
    }

    @tf.function
    def parser_fn(single_record):
        parsed_record = tf.io.parse_single_example(single_record,
                                                   feature_description)
        return parsed_record
    return parser_fn


def tf_example_uniform_sampler(paths: List[utils.PathStr],
                               num_map_threads: int, number_to_sample: int,
                               parser_fn: Callable):
    """
    paths:
        Paths of the TFRecord files from which to sample
    num_map_threads:
        Number of threads
    """
    if not paths:
        raise ValueError("Didn't receive any paths to read from.")
    if not num_map_threads > 0:
        raise ValueError(num_map_threads)
    if not number_to_sample > 0:
        raise ValueError(number_to_sample)

    cache = {}

    logging.info("\nDo not believe the following tqdm timer's duration "
                 "estimation, caching messes the evaluation.")
    for _ in tqdm.tqdm(range(number_to_sample)):
        path = random.choice(paths)
        if path not in cache:
            d = tf.data.Dataset.from_tensor_slices([path])
            d = d.interleave(tf.data.TFRecordDataset,
                             num_parallel_calls=tf.data.experimental.AUTOTUNE)
            cache[path] = list(d)

        yield parser_fn(random.choice(cache[path]))


def read_from_tf_example(paths: List[utils.PathStr], sample_len: int,
                         num_epochs: int, num_map_threads: int,
                         parser_fn: Callable,
                         sharding_idx: Optional[int], sharding_quantity: int,
                         shuffle_buffer_size: int = 1,
                         ) -> tf.data.Dataset:
    """Creates a Tensorflow parallel dataset reader for the TfExamples.
  Also could support sharding (meaning, separating the dataset over shards), as
  well as many different features of tf.data

  Arguments:  
    paths: Paths of the tf.Example files.
    sample_len: Length in number of tokens of the sample.
    num_epochs: Number of times to loop over the data.
    num_map_threads: Number fo threads to use for the deserialization
                     of the tf.Example bytes to Python (Tensorflow) objects.
    parser_fn: Function to extract a tf.Feature from a tf.Example entry
    shuffle_buffer_size: the data is loaded this number of samples at the time,
                         and these "shuffle batches" have their sample shuffled.
    sharding_idx:
    sharding_quantity:
  Returns:
    A tf.data.Dataset object that returns the samples one at the time.
  """
    if not paths:
        raise ValueError("Didn't receive any paths to read from.")
    if not sample_len > 0:
        raise ValueError(sample_len)
    if not num_map_threads > 0:
        raise ValueError(num_map_threads)

    paths = [str(path) for path in paths]

    d = tf.data.Dataset.from_tensor_slices(paths)
    if sharding_quantity and sharding_quantity > 1:
        d = d.shard(sharding_quantity, sharding_idx)
    d = d.repeat(num_epochs)
    d = d.interleave(tf.data.TFRecordDataset,
                     num_parallel_calls=tf.data.experimental.AUTOTUNE)
    d = d.shuffle(shuffle_buffer_size)

    return d.map(parser_fn, num_parallel_calls=num_map_threads)


if __name__ == "__main__":
    social_iqa_path = (pathlib.Path(__file__).resolve().parent /
                       "splitted-socialiqa.example")
    assert social_iqa_path.exists(), "Call the example script first"
    reader = read_from_tf_example([social_iqa_path], 128, num_epochs=2,
                                  shuffle_buffer_size=1, num_map_threads=1)

    print(np.stack(list(map(lambda x: x["input_ids"].numpy(), reader))).shape)
