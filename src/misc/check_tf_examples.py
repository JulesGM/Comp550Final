# ============================================================================
# Check that the pre-training tf-example files are in good shape
#
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


def read_tf_file(args, cur_file_path):
    # Read the file
    reader = tf_example_utils.read_from_tf_example(
        paths=[cur_file_path],
        sample_len=args.sample_length,
        shuffle_buffer_size=args.shuffle_buffer_size,
        num_map_threads=args.num_map_threads,
        sharding_idx=args.sharding_idx,
        sharding_quantity=args.sharding_quantity,
        num_epochs=1,
        parser_fn=tf_example_utils.build_filter_input_parser_fn(
            args.sample_length)
    )

    # Iterate and count
    num_datapts = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        next_getter = reader.batch(args.batch_size).make_one_shot_iterator().get_next()

        try:
            while True:
                data = sess.run(next_getter)
                # first dim of input_ids should be the current batch size
                num_datapts += len(data['input_ids'])

        except tf.errors.DataLossError as e:
            print(e)
            # Return
            return num_datapts, False

        except tf.errors.OutOfRangeError:
            pass
    # Return
    return num_datapts, True


def main(args):
    # ==
    # Keep track of variables
    data_dict = {
        'file_name': [],
        'num_data_pts': [],
        'is_good': []
    }

    # ==
    # Print-outs and get input paths
    print(f"Input data path: {str(args.input_dir_path)}")

    in_files = glob.glob(
        os.path.join(args.input_dir_path, args.file_glob)
    )
    in_files.sort()
    print(f"\nNumber of files found: {len(in_files)}\n")

    # ==
    # Iterate over each input file to find error


    # Iterate
    for i, cur_file in enumerate(in_files):
        print(f"{i}\t{cur_file.split('/')[-1]}")

        # Check
        num_data_pts, isgood = read_tf_file(args, cur_file)

        print(num_data_pts, isgood)

        # Log
        data_dict['file_name'].append(cur_file.split('/')[-1])
        data_dict['num_data_pts'].append(num_data_pts)
        data_dict['is_good'].append(isgood)


    # ==
    # Final evaluation

    df = pd.DataFrame.from_dict(data_dict)

    print("\n==========")
    print(df)
    print("==========\n")

    # Print just the files with good data
    print("\n==========")
    df_filtered = df[df['is_good']==True]
    data_pts_list = df_filtered['num_data_pts'].values
    for j, file_name in enumerate(df_filtered['file_name'].values):
        print(file_name, data_pts_list[j])
    print("==========\n")

    print("\n==========")
    print(f"Files checked: {len(df)}")
    print(f"Files in good condition: {sum(df['is_good'].values)}")
    print("==========\n")



if __name__ == "__main__":
    # ==
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir_path", type=str, required=True,
                        help="Path to the data directory.")
    parser.add_argument("--file_glob", type=str, default='*tfrecord', # TODO CHANGE ME
                        help="Glob pattern to find the files")

    parser.add_argument("--num_map_threads", type=int, default=2,
                        help="Number of threads to use to de-serialize the "
                             "dataset.")

    parser.add_argument("--shuffle_buffer_size", type=int, default=1,
                        help=("shuffle_buffer_size for tf.data.Dataset of "
                              "the main data loader."))
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Size of the mini-batches.")
    parser.add_argument("--sample_length", type=int, default=128,
                        help="Sample length")
    parser.add_argument("--sharding_quantity", type=int, default=0)
    parser.add_argument("--sharding_idx", type=int, default=None)

    args = parser.parse_args()
    print(args)

    # ==
    # Run rest of script
    main(args)
