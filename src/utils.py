import logging
import pathlib
from typing import Union, Iterable, Type
from urllib import request
import zipfile

PathStr = Union[str, pathlib.Path]

def get_filename_from_url(url: str) -> str:
    """ Get the filename from a URL we are likely meant to download a file from. 
    """
    return url.rsplit("/", 1)[1]


def maybe_download(url: str, output_path: PathStr, force: bool=False) -> None:
    """ Download if no file with the same name already exists at `output_path`.
    """
    output_path = pathlib.Path(output_path)
    if force or not output_path.exists():
        logging.info("Downloading")
        request.urlretrieve(url, output_path)
    

def maybe_unzip(path_to_zip: PathStr, output_folder: PathStr,
                force: bool=False) -> None:
    """ Maybe the file at `path_to_zip` to `output_folder`
    
    Unzips the file if there isn't a file with the same name in `output_folder`.
    
    TODO(julesgm): This function needs a bit of testing.
    """
    path_to_zip = pathlib.Path(path_to_zip)
    output_folder = pathlib.Path(output_folder)

    assert str(path_to_zip).endswith(".zip"), path_to_zip

    if force or not (output_folder/path_to_zip.name.split(".")[0]).exists():
        logging.info(f"Unzipping to {output_folder}")
        with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
            zip_ref.extractall(output_folder)
    
    
def maybe_download_and_unzip(url: str, output_folder: PathStr=None, 
                             save_zip_where: PathStr=None, 
                             force: bool=False) -> None:
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
    
    # Don't redownload the zip if the unzipped version already exists
    if force or not (output_folder/filename.split(".")[0]).exists():
        logging.info(f"Maybe downloading {url}")
        maybe_download(url, save_zip_where/filename, force)

        logging.info(f"Maybe unzipping {filename}")
        maybe_unzip(save_zip_where/filename, output_folder, force)

def check_type(obj, types: Union[Iterable[Type], Type]):
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