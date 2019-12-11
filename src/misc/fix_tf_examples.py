# ============================================================================
# Fix the pretraining tf-example files with Data Loss Error
#
# ============================================================================

import argparse
import glob
import os
import pathlib
import sys

import pandas as pd
import tensorflow as tf

sys.path.append('./')
import tf_example_utils
import utils


def readwrite_one_tf_file(args, reader, writer):
    # ==
    # iterate over input file and write to output file
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        next_getter = reader.batch(args.batch_size).make_one_shot_iterator().get_next()

        try:
            while True:
                # Try to get the data batch
                data = sess.run(next_getter)

                # If successful, write
                writer.from_masked_feature_batch(
                    data,
                    max_predictions_per_seq=args.max_predictions_per_seq
                )

        except tf.errors.OutOfRangeError:
            pass

        # If dataloss error, return
        except tf.errors.DataLossError as e:
            print(e)
            return


def readwrite_tf_files(args, in_files, writer):
    # ==
    # Iterate over each file
    for i, cur_file_path in enumerate(in_files):
        print(f"[{i + 1}/{len(in_files)}]\t{cur_file_path.split('/')[-1]}")

        # ==
        # Read the file
        reader = tf_example_utils.read_from_tf_example(
            paths=[cur_file_path],
            sample_len=args.sequence_length,
            shuffle_buffer_size=args.shuffle_buffer_size,
            num_map_threads=args.num_map_threads,
            sharding_idx=args.sharding_idx,
            sharding_quantity=args.sharding_quantity,
            num_epochs=1,
            parser_fn=tf_example_utils.build_masked_parser_fn(
                args.sequence_length, args.max_predictions_per_seq)
        )

        # ==
        # Iterate and write file
        readwrite_one_tf_file(args, reader, writer)

        print()


def main(args):
    # ==
    # Get input paths and print
    print(f"Input data path: {str(args.input_dir_path)}")

    in_files = glob.glob(
        os.path.join(args.input_dir_path, args.file_glob)
    )
    in_files.sort()
    print(f"\nNumber of files found: {len(in_files)}\n")

    # ==
    # Generate list of output files
    output_files = []
    for tf_ex_idx in range(args.num_out_files):
        cur_tf_file_name = f"{tf_ex_idx}_BertExample.tfrecord"
        output_files.append(
            os.path.join(args.output_dir_path, cur_tf_file_name)
        )

    # ==
    # Start example writer
    with tf_example_utils.BERTExampleWriter(
            output_files=output_files,
            vocab_path=args.vocab_file_path,
            max_num_tokens=args.sequence_length) as writer:

        readwrite_tf_files(args, in_files, writer)


if __name__ == "__main__":
    # ==
    # Parse arguments

    # TODO check these out

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir_path", type=str, required=True,
                        help="Path to the data directory.")
    parser.add_argument("--output_dir_path", type=str, required=True,
                        help="Number of output tf-example files to write")
    parser.add_argument("--vocab_file_path", type=str, required=True,
                        help="Path to the BERT vocabulary file.")

    parser.add_argument("--file_glob", type=str, default='*tfrecord',
                        help="Glob pattern to find the input files")
    parser.add_argument("--num_out_files", type=int, default=100,
                        help="Number of output tf-example files to write")

    parser.add_argument("--sequence_length", type=int, default=128,
                        help="Sequence length / max number tokens (Default: 128)")
    parser.add_argument("--max_predictions_per_seq", type=int, default=20,
                        help="Max prediction per seq")

    parser.add_argument("--num_map_threads", type=int, default=2,
                        help="Number of threads to use to de-serialize the "
                             "dataset.")
    parser.add_argument("--shuffle_buffer_size", type=int, default=1,
                        help=("shuffle_buffer_size for tf.data.Dataset of "
                              "the main data loader."))
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Size of the mini-batches.")

    parser.add_argument("--sharding_quantity", type=int, default=0)
    parser.add_argument("--sharding_idx", type=int, default=None)

    args = parser.parse_args()
    print(args)

    # ==
    # Run rest of script
    main(args)
