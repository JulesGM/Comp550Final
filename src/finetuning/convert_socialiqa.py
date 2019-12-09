# -*- coding: utf-8 -*-
"""
"""

import argparse
import os
import json
import logging
from typing import Iterable

import tqdm

COMMONSENSE_TEMPLATE = {
      "answerKey": None, 
      "id": None, 
      "question": 
        {
          "stem": None,
          "choices": [
            {
              "label": "A", 
              "text": None
            }, 
            {
              "label": "B", 
              "text": None
            }, 
            {
              "label": "C", 
              "text": None
            }, 
          ]
      }
    }

def traversal_test(node: Iterable) -> None:
  """Make sure we haven't forgotten to fill in a value of the sample template.
  """
  if type(node) == dict:
    node = node.values()

  for child in node:
    if type(child) in {list, dict, set}:
      traversal_test(child)
    if child is None:
      raise("Found a None value")

def main(args: argparse.Namespace):
  file_names = dict(train="socialIQa_v1.4_trn.jsonl",
                    dev="socialIQa_v1.4_dev.jsonl",
                    test="socialIQa_v1.4_tst.jsonl",)

  for cv_set in ["train", "dev", "test"]:
    input_path = os.path.join(args.input_dir, file_names[cv_set])
    output_path = os.path.join(args.output_dir, f"socialIQa_{cv_set}.jsonl")
    with open(input_path, 'r') as fin, open(output_path, "w") as fout:
      samples = [json.loads(line) for line in fin]
      for index, social in enumerate(tqdm.tqdm(samples)):
        # Copy the template
        test_new_social = dict(COMMONSENSE_TEMPLATE) 
        test_new_social["id"]= str(index) 
        test_new_social["context"] = social["context"]
        test_new_social["answerKey"] = social["correct"]
        test_new_social["question"]["stem"] = social["question"]

        for answers in test_new_social["question"]["choices"]:
          answers["text"] = social[f"answer{answers['label']}"]

        # Make sure we haven't forgotten anything
        traversal_test(test_new_social)
        
        # Write
        json.dump(test_new_social, fout)
        fout.write("\n")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("input_dir")
  parser.add_argument("output_dir")
  args = parser.parse_args()
  main(args)