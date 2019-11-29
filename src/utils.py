import argparse
import itertools
import logging
import pathlib
from typing import Any, List, Optional, Iterable, \
                   Tuple, Type, TypeVar, Union
from urllib import request
import zipfile

import numpy as np

PathStr = Union[str, pathlib.Path]

def get_filename_from_url(url: str) -> str:
    """ Get the filename from a URL we are likely meant to download a file from. 
    """
    return url.rsplit("/", 1)[1]


def maybe_download(url: str, output_path: PathStr, 
                   force: bool = False) -> None:
    """ Download if no file with the same name already exists at `output_path`.
    """
    output_path = pathlib.Path(output_path)
    already_there = output_path.exists()
    if already_there:
        logging.debug(f"maybe_download: File {output_path} was already"
                      f"there. Got force={force}")

    if force or not already_there:
        logging.info("maybe_download: Downloading")
        request.urlretrieve(url, output_path)
    

def maybe_unzip(path_to_zip: PathStr, output_folder: PathStr,
                force: bool = False) -> None:
    """ Maybe the file at `path_to_zip` to `output_folder`
    
    Unzips the file if there isn't a file with the same name in `output_folder`.
    
    TODO(julesgm): This function needs a bit of testing.
    """
    path_to_zip = pathlib.Path(path_to_zip)
    output_folder = pathlib.Path(output_folder)

    assert str(path_to_zip).endswith(".zip"), path_to_zip

    final_path = output_folder/path_to_zip.name.split(".")[0]
    already_there = final_path.exists()
    if already_there:
        logging.debug(f"maybe_unzip: File {path_to_zip} was already there. Got force={force}")

    if force or not already_there:
        logging.info(f"maybe_unzip: Unzipping to {output_folder}")
        with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
            zip_ref.extractall(output_folder)

    
def maybe_download_and_unzip(url: str, output_folder: PathStr = None, 
                             save_zip_where: PathStr = None, 
                             force: bool = False) -> None:
    """ Download the file from url & unzip it if necessary.
    
    Downloads the file if there isn't alread a file in `output_folder` with 
    the same name, zipped or unzipped. Unzips the file if it downloads it.
    """
    if not output_folder:
        output_folder = pathlib.Path.cwd().resolve()
    output_folder = pathlib.Path(output_folder)

    if save_zip_where:
        save_zip_where = pathlib.Path(save_zip_where)
    else:
        save_zip_where = output_folder
    filename = get_filename_from_url(url)
    
    final_path = output_folder/filename.split(".")[0]
    already_there = final_path.exists()
    if already_there:
        logging.debug(f"maybe_download_and_unzip: File {final_path} "
                      f"was already there. Got force={force}")

    # Don't redownload the zip if the unzipped version already exists
    if force or not already_there:
        logging.info(f"maybe_download_and_unzip: "
                     f"Maybe downloading {url}")
        maybe_download(url, save_zip_where/filename, force)

        logging.info(f"maybe_download_and_unzip: "
                     f"Maybe unzipping {filename}")
        maybe_unzip(save_zip_where/filename, output_folder, force)


def check_type(obj: Any, types: Union[Iterable[Type], Type]
               ) -> None:
    """Check if an object is one of a few possible types.
    """ 
    if not hasattr(types, "__iter__"):
        types = [types]

    fit_one = any(isinstance(obj, type_) for type_ in types)
    if not fit_one:
        raise RuntimeError(f"Expected object to be one of the following types:"
                           f"{types}. "
                           f"Got type {type(obj)} instead, which is not "
                           f"an instance of it.")


def log_args(args: argparse.Namespace, log_level: int = 
             int(logging.DEBUG)) -> None:
    """Logs the contents of a Namespace object in a pretty way.
    """
    before_entry = "\n\t- "
    logging.log(log_level, 
                f"Args:{before_entry}" + before_entry.join(
                    f"{k}: {v}" for k, v in vars(args).items()))


def count_len_iter(iterable: Iterable, 
                   force_count: bool = False) -> int:
    """Counts the items of an iterable. May not return if the iterable has no end.
    Some iterables (like text files) don't have an easy way to query their 
    length, you need to actually count the elements.
    `force_count` forces the function to actually count the elements instead of just
    callen len.
    """
    # Check if it has a function to compute the len
    if hasattr(iterable, "__len__") and not force_count:
        return len(iterable)

    # Iterate and add one per iteration
    return sum(1 for _ in iterable)


def count_lines(path: PathStr) -> int:
    """Counts the number of lines in a file
    """
    with open(path) as fin:
        return count_len_iter(fin)


def to_categorical(y : Iterable, num_classes: int = None, 
                   dtype="float32") -> np.ndarray:
    """
    Copy pasted from tf.keras.utils to not have this huge
    dependency.
    
    Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    Arguments:
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input. Default: `'float32'`.
    Returns:
        A binary matrix representation of the input. The classes axis is placed
        last.
    """
    y = np.array(y, dtype='int')
    input_shape: Tuple = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def check_equal(obj_a: Any, obj_b: Any, 
                ErrorType: Type = ValueError) -> None:
    if not obj_a == obj_b:
        raise ErrorType("`check_equal` failed. Got:\n"
                        f"\t{obj_a} and\n"
                        f"\t{obj_b}.")


T = TypeVar("T")
def grouper(n: int, iterable: Iterable[T], 
            fillvalue: Optional[T] = None, 
            mode: str = "longest") -> Iterable[List[T]]:
    """Chunks iterables in packs of n. 
    
    Examples: 
        grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
        grouper(3, 'ABCDEFG', mode="shortest") --> ABC DEF 
    
    Slight modification of the `grouper` recipe of 
    https://docs.python.org/3/library/itertools.html#itertools-recipes
    to support mode="shortest".
    
    """
    
    acceptable_values = {"longest", "shortest"}
    if mode not in acceptable_values:
        raise ValueError(f"Argument 'mode' should be one of "
                         f"{acceptable_values}. Got '{mode}' instead.")
    
    args = [iter(iterable)] * n
    
    if mode == "longest":
        return itertools.zip_longest(fillvalue=fillvalue, *args)

    elif mode == "shortest":
        return zip(*args)

    assert False, "Should never get this far."


def safe_bool_arg(arg: str) -> bool:
    """Argparse's builtin handling of strings is horrendous. We make it safer.
    Meant to be used as the `type` argument of `parser.add_argument` for a bool.

    Arguments:
        arg: The command line string to parse as a bool.
    Returns:
        The parsed argument
    """
    positive_values = {"true", "t", "yes", "y"}
    negative_values = {"false", "f", "no", "n"}
    if arg.lower() in positive_values:
        return True
    elif arg.lower() in negative_values:
        return False
    else:
        raise ValueError(f"Got a wrong value for a boolean arg. Experted one of "
                         f"{positive_values + negative_values}. Got {arg}.")


if __name__ == "__main__":
    """These are tests for the utils.
    """
    FORMAT = '%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # log_args:
    args = argparse.Namespace(potato="apple", addresses=["mars", "saturn"], 
                              phone_number=31415926535)
    log_args(args)
    
    # count_len_iter:
    # with __len__
    assert count_len_iter([123, 321, 22]) == 3
    # without __len__
    assert count_len_iter((x for x in range(33))) == 33
