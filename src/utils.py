import logging
import pathlib
from typing import Union
from urllib import request
import zipfile

PathStr = Union[str, pathlib.Path]

def get_filename_from_url(url: str) -> str:
    """ Get the filename from a URL we are likely meant to download a file from. 
    """
    return url.rsplit("/", 1)[1]


def maybe_download(url: str, output_path: PathStr) -> None:
    """ Download if no file with the same name already exists at `output_path`.
    """
    output_path = pathlib.Path(output_path)
    if not output_path.exists():
        logging.info("Downloading")
        request.urlretrieve(url, output_path)
    

def maybe_unzip(path_to_zip: PathStr, output_folder: PathStr) -> None:
    """ Maybe the file at `path_to_zip` to `output_folder`
    
    Unzips the file if there isn't a file with the same name in `output_folder`.
    
    TODO(julesgm): This function needs a bit of testing.
    """
    path_to_zip = pathlib.Path(path_to_zip)
    output_folder = pathlib.Path(output_folder)

    assert str(path_to_zip).endswith(".zip"), path_to_zip

    if not (output_folder/path_to_zip.name.split(".")[0]).exists():
        logging.info(f"Unzipping to {output_folder}")
        with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
            zip_ref.extractall(output_folder)
    
    
def maybe_download_and_unzip(url: str, output_folder: PathStr="") -> None:
    """ Download the file from url & unzip it if necessary.
    
    Downloads the file if there isn't alread a file in `output_folder` with 
    the same name, zipped or unzipped. Unzips the file if it downloads it.
    """
    output_folder = pathlib.Path(output_folder)
    filename = get_filename_from_url(url)
    
    # Don't redownload the zip if the unzipped version already exists
    if not (output_folder/filename.split(".")[0]).exists():
        logging.info(f"Maybe downloading {url}")
        maybe_download(url, filename)

        logging.info(f"Maybe unzipping {filename}")
        maybe_unzip(filename, output_folder)