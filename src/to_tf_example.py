import collections
import pathlib
import subprocess
from typing import Any, Iterable, List, Tuple, Type, Union

import random
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


def _create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
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

        next_sentence_label = 1 if b_is_random else 0

        features = collections.OrderedDict()
        features["input_ids"] = _create_int_feature(bpe_ids)
        features["segment_ids"] = _create_int_feature(segment_ids)
        features["next_sentence_labels"] = _create_int_feature([next_sentence_label])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        self._writers[self._writer_index].write(tf_example.SerializeToString())
        self._writer_index = (self._writer_index + 1) % len(self._writers)

    def close(self):
        for writer in self._writers:
            writer.close()


if __name__ == "__main__":
    # Basic test code
    subprocess.check_output(["wget", 
        "https://raw.githubusercontent.com/microsoft/BlingFire/master/ldbsrc/bert_base_cased_tok/vocab.txt"])

    with WriteAsTfExample(list(), "vocab.txt", 128) as writer:
        pass
