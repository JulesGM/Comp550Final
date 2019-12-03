import collections
import pathlib
import random
import subprocess
from typing import Any, Dict, Iterable, List, Tuple, Type, Union

import numpy as np
try:
  import colored_traceback.auto
except ImportError:
  pass
import tensorflow as tf

import utils


CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"

def _truncate_seq_pair(tokens_a: list, tokens_b: list, max_num_tokens: int, rng
                      ) -> None:
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
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=feature_list))
  return feature


class WriteAsTfExample:
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
    def __init__(self, output_files: List[utils.PathStr], vocab_path: utils.PathStr, 
                 max_num_tokens: int):
        """Multiple output_files for if we want to save in multiple files.
        """
        vocab_path = pathlib.Path(vocab_path)
        output_files = [pathlib.Path(output_path) for output_path in output_files]

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

    def __exit__(self, type, value, tb):
        self.close()
        
    def _write_one(self, feature_dict):
      tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
      self._writers[self._writer_index].write(tf_example.SerializeToString())
      self._writer_index = (self._writer_index + 1) % len(self._writers)

    def from_feature_batch(self, batch):
      """
      Weird dict unbatching.
      """
      if not (len(batch["input_ids"]) == len(batch["input_mask"]) 
              == len(batch["segment_ids"])):
        raise RuntimeError("One feature had a weird length.")

      for input_ids, input_mask, segment_ids, next_sentence_labels in zip(
        batch["input_ids"], batch["input_mask"], batch["segment_ids"], 
        batch["next_sentence_labels"]):

        features = collections.OrderedDict()
        features["input_ids"] = _create_int_feature(input_ids, 
          self._max_num_tokens)
        features["input_mask"] = _create_int_feature(input_mask, 
          self._max_num_tokens)
        features["segment_ids"] = _create_int_feature(segment_ids, 
          self._max_num_tokens)
        features["next_sentence_labels"] = _create_int_feature([
          next_sentence_labels], 1)

        self._write_one(features)
      
      # In one beautiful // terrible line. Works because dicts are ordered now:
      # [self._write_one({k: v for k, v in zip(batch, vs)}) for vs in zip(*batch.values())]
      

    def add_sample(self, bpe_ids_a: List[int], bpe_ids_b: List[int], 
                   b_is_random: bool) -> None:
        """Convert to tf.Example, then write to a file.
        """

        # We copy the lists because truncate_seq_pairs modifies them
        bpe_ids_a = list(bpe_ids_a)
        bpe_ids_b = list(bpe_ids_b)
        
        # -3 Because of the CLS token and of the two SEP tokens
        _truncate_seq_pair(bpe_ids_a, bpe_ids_b, self._max_num_tokens - 3, random)
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
        features["input_ids"] = _create_int_feature(bpe_ids, self._max_num_tokens)
        features["input_mask"] = _create_int_feature(segment_ids, self._max_num_tokens)
        features["segment_ids"] = _create_int_feature(input_mask, self._max_num_tokens)
        features["next_sentence_labels"] = _create_int_feature([next_sentence_label], 1)

        self._write_one(features)

    def close(self):
        for writer in self._writers:
            writer.close()

def readFromTfExample(paths: List[utils.PathStr], sample_len: int, 
                      num_epochs: int, num_map_threads: int,
                      shuffle_buffer_size: int = 1) -> tf.data.Dataset:
  """Creates a Tensorflow parallel dataset reader for the TfExamples.
  Also could support sharding (meaning, seperating the dataset over shards), as
  well as many different features of tf.data

  Arguments:  
    paths: Paths of the tf.Example files.
    sample_len: Length in number of tokens of the sample.
    num_epochs: Number of times to loop over the data.
    num_map_threads: Number fo threads to use for the deserialization
                     of the tf.Example bytes to Python (Tensorflow) objects.
    shuffle_buffer_size: the data is loaded this number of samples at the time,
                         and these "shuffle batches" have their sample shuffled.
  Returns:
    A tf.data.Dataset object that returns the samples one at the time.
  """
  paths = [str(path) for path in paths]

  _feature_description = {
    "input_ids": tf.io.FixedLenFeature([sample_len], tf.int64,),
    "input_mask": tf.io.FixedLenFeature([sample_len], tf.int64,),
    "segment_ids": tf.io.FixedLenFeature([sample_len], tf.int64,),
    "next_sentence_labels": tf.io.FixedLenFeature([1], tf.int64,)
    } 

  # This just jits the function for the tf.data.Dataset graph
  @tf.function
  def _parser_fn(single_record):
    parsed_record = tf.io.parse_single_example(single_record, _feature_description)
    return parsed_record

  d = tf.data.Dataset.from_tensor_slices(paths)
  d = d.repeat(num_epochs)
  d = d.interleave(tf.data.TFRecordDataset, block_length=1
                  )
  d = d.shuffle(shuffle_buffer_size)
                      
  return d.map(_parser_fn, num_parallel_calls=num_map_threads)

if __name__ == "__main__":
    social_iqa_path = (pathlib.Path(__file__).resolve().parent/
                       "splitted-socialiqa.example")
    assert social_iqa_path.exists(), "Call the example script first"
    reader = readFromTfExample([social_iqa_path], 128, num_epochs=2, shuffle_buffer_size=1,
                               num_map_threads=1)

    print(np.stack(list(map(lambda x: x["input_ids"].numpy(), reader))).shape)
    # for x in reader:
    #   print(x)
                               