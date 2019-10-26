import logging
import pathlib
from typing import Union
from urllib import request
import zipfile

PathStr = Union[str, pathlib.Path]

def get_filename_from_url(url: str) -> str:
    return url.rsplit("/", 1)[1]


def maybe_download(url: str, output_path: PathStr) -> None:
    output_path = pathlib.Path(output_path)
    if not output_path.exists():
        logging.info("Downloading")
        request.urlretrieve(url, output_path)
    

def maybe_unzip(path_to_zip: PathStr, output_folder: PathStr) -> None:
    """ Maybe the file at `path_to_zip` to `output_folder`
    Unzips the file if there isn't a file with the same name in `output_folder`.
    """
    path_to_zip = pathlib.Path(path_to_zip)
    output_folder = pathlib.Path(output_folder)
    assert str(path_to_zip).endswith(".zip"), path_to_zip

    if not (output_folder/path_to_zip.name.split(".")[0]).exists():
        logging.info(f"Unzipping to {output_folder}")
        with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
            zip_ref.extractall(output_folder)
    
    
def maybe_download_and_unzip(url: str, output_folder: str=None) -> None:
    filename = get_filename_from_url(url)
    
    logging.info(f"Maybe downloading {url}")
    maybe_download(url, filename)

    logging.info(f"Maybe unzipping {filename}")
    maybe_unzip(filename, "")