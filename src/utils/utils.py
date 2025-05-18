"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import csv
import functools
import io
import json
import logging
import os
import pickle
import re
import shutil
import urllib
import urllib.error
import urllib.request
from io import BytesIO
from typing import Any
from typing import List, Dict
from typing import Optional, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import torch
import yaml
from PIL import Image
from iopath.common.download import download
from iopath.common.file_io import file_lock, g_pathmgr
from src.common.registry import registry
from torch import nn
from torch.utils.model_zoo import tqdm
from torchvision.datasets.utils import (
    check_integrity,
    download_file_from_google_drive,
    extract_archive,
)

import dist_utils


def neg_clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative CLIP loss by combining the caption and image loss.
    """
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t()[:len(similarity)])
    return (caption_loss + image_loss) / 2.0


def load_and_resize_image(url: str, size=(256, 256)) -> Image:
    try:
        response = requests.get(url, timeout=5)  # 타임아웃 설정
        response.raise_for_status()  # HTTP 에러 발생 시 예외 처리
        img = Image.open(BytesIO(response.content))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_resized = img.resize(size)  # 리사이즈 추가
        return img_resized
    except Exception:
        return None  # 예외 발생 시 None 반환


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/2021-03-07-clip.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    """
    This method is equivalent to tensor.norm(p=2, dim=-1, keepdim=True) and used to make
    model `executorch` exportable. See issue https://github.com/pytorch/executorch/issues/3566
    """
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    normed_tensor = torch.pow(sum_tensor, 0.5)
    return normed_tensor


def now():
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d%H%M")[:-1]


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def get_cache_path(rel_path):
    return os.path.expanduser(os.path.join(registry.get_path("cache_root"), rel_path))


def get_abs_path(rel_path):
    return os.path.join(registry.get_path("library_root"), rel_path)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def load_yml(filename):
    with open(filename, "r") as f:
        return yaml.safe_load(f)


def load_tsv(filename) -> list[list[str]]:
    with open(filename, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        data = [row for row in reader]
    return data


# The following are adapted from torchvision and vissl
# torchvision: https://github.com/pytorch/vision
# vissl: https://github.com/facebookresearch/vissl/blob/main/vissl/utils/download.py


def makedir(dir_path):
    """
    Create the directory if it does not exist.
    """
    is_success = False
    try:
        if not g_pathmgr.exists(dir_path):
            g_pathmgr.mkdirs(dir_path)
        is_success = True
    except BaseException:
        print(f"Error creating directory: {dir_path}")
    return is_success


def get_redirected_url(url: str):
    """
    Given a URL, returns the URL it redirects to or the
    original URL in case of no indirection
    """
    import requests

    with requests.Session() as session:
        with session.get(url, stream=True, allow_redirects=True) as response:
            if response.history:
                return response.url
            else:
                return url


def to_google_drive_download_url(view_url: str) -> str:
    """
    Utility function to transform a view URL of Google Drive
    to a download URL for Google Drive
    Example input:
        https://drive.google.com/file/d/137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp/view
    Example output:
        https://drive.google.com/uc?export=download&id=137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp
    """
    splits = view_url.split("/")
    assert splits[-1] == "view"
    file_id = splits[-2]
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def download_google_drive_url(url: str, output_path: str, output_file_name: str):
    """
    Download a file from Google Drive
    Downloading a URL from Google Drive requires confirmation when
    the file of the size is too big (Google Drive notifies that
    anti-viral checks cannot be performed on such files)
    """
    import requests

    with requests.Session() as session:

        # First get the confirmation token and append it to the URL
        with session.get(url, stream=True, allow_redirects=True) as response:
            for k, v in response.cookies.items():
                if k.startswith("download_warning"):
                    url = url + "&confirm=" + v

        # Then download the content of the file
        with session.get(url, stream=True, verify=True) as response:
            makedir(output_path)
            path = os.path.join(output_path, output_file_name)
            total_size = int(response.headers.get("Content-length", 0))
            with open(path, "wb") as file:
                from tqdm import tqdm

                with tqdm(total=total_size) as progress_bar:
                    for block in response.iter_content(
                            chunk_size=io.DEFAULT_BUFFER_SIZE
                    ):
                        file.write(block)
                        progress_bar.update(len(block))


def _get_google_drive_file_id(url: str) -> Optional[str]:
    parts = urlparse(url)

    if re.match(r"(drive|docs)[.]google[.]com", parts.netloc) is None:
        return None

    match = re.match(r"/file/d/(?P<id>[^/]*)", parts.path)
    if match is None:
        return None

    return match.group("id")


def _urlretrieve(url: str, filename: str, chunk_size: int = 1024) -> None:
    with open(filename, "wb") as fh:
        with urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "vissl"})
        ) as response:
            with tqdm(total=response.length) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    fh.write(chunk)


def download_url(
        url: str,
        root: str,
        filename: Optional[str] = None,
        md5: Optional[str] = None,
) -> Union[str | None | Any]:
    """Download a file from an url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under.
                                  If None, use the basename of the URL.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir(root)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
        return

    # expand redirect chain if needed
    url = get_redirected_url(url)

    # check if file is located on Google Drive
    file_id = _get_google_drive_file_id(url)
    if file_id is not None:
        return download_file_from_google_drive(file_id, root, filename, md5)

    # download the file
    try:
        print("Downloading " + url + " to " + fpath)
        _urlretrieve(url, fpath)
    except (urllib.error.URLError, IOError) as e:  # type: ignore[attr-defined]
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            print(
                "Failed download. Trying https -> http instead."
                " Downloading " + url + " to " + fpath
            )
            _urlretrieve(url, fpath)
        else:
            raise e

    # check integrity of downloaded file
    if not check_integrity(fpath, md5):
        raise RuntimeError("File not found or corrupted.")

    return fpath


def download_and_extract_archive(
        url: str,
        download_root: str,
        extract_root: Optional[str] = None,
        filename: Optional[str] = None,
        md5: Optional[str] = None,
        remove_finished: bool = False,
) -> None:
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print("Extracting {} to {}".format(archive, extract_root))
    extract_archive(archive, extract_root, remove_finished)


def cache_url(url: str, cache_dir: str) -> str:
    """
    This implementation downloads the remote resource and caches it locally.
    The resource will only be downloaded if not previously requested.
    """
    parsed_url = urlparse(url)
    dirname = os.path.join(cache_dir, os.path.dirname(parsed_url.path.lstrip("/")))
    makedir(dirname)
    filename = url.split("/")[-1]
    cached = os.path.join(dirname, filename)
    with file_lock(cached):
        if not os.path.isfile(cached):
            logging.info(f"Downloading {url} to {cached} ...")
            cached = download(url, dirname, filename=filename)
    logging.info(f"URL {url} cached in {cached}")
    return cached


def create_file_symlink(file1, file2):
    """
    Simply create the symlinks for a given file1 to file2.
    Useful during model checkpointing to symlinks to the
    latest successful checkpoint.
    """
    try:
        if g_pathmgr.exists(file2):
            g_pathmgr.rm(file2)
        g_pathmgr.symlink(file1, file2)
    except Exception as e:
        logging.info(f"Could NOT create symlink. Error: {e}")


def save_file(data, filename, append_to_json=True, verbose=True):
    """
    Common i/o utility to handle saving data to various file formats.
    Supported:
        .pkl, .pickle, .npy, .json
    Specifically for .json, users have the option to either append (default)
    or rewrite by passing in Boolean value to append_to_json.
    """
    if verbose:
        logging.info(f"Saving data to file: {filename}")
    file_ext = os.path.splitext(filename)[1]
    if file_ext in [".pkl", ".pickle"]:
        with g_pathmgr.open(filename, "wb") as f_open:
            pickle.dump(data, f_open, pickle.HIGHEST_PROTOCOL)
    elif file_ext == ".npy":
        with g_pathmgr.open(filename, "wb") as f_open:
            np.save(f_open, data)
    elif file_ext == ".json":
        if append_to_json:
            with g_pathmgr.open(filename, "a") as f_open:
                f_open.write(json.dumps(data, sort_keys=True) + "\n")
                f_open.flush()
        else:
            with g_pathmgr.open(filename, "w") as f_open:
                f_open.write(json.dumps(data, sort_keys=True) + "\n")
                f_open.flush()
    elif file_ext == ".yaml":
        with g_pathmgr.open(filename, "w") as f_open:
            dump = yaml.dump(data)
            f_open.write(dump)
            f_open.flush()
    else:
        raise Exception(f"Saving {file_ext} is not supported yet")

    if verbose:
        logging.info(f"Saved data to file: {filename}")


def load_file(filename, mmap_mode=None, verbose=True, allow_pickle=False):
    """
    Common i/o utility to handle loading data from various file formats.
    Supported:
        .pkl, .pickle, .npy, .json
    For the npy files, we support reading the files in mmap_mode.
    If the mmap_mode of reading is not successful, we load data without the
    mmap_mode.
    """
    if verbose:
        logging.info(f"Loading data from file: {filename}")

    file_ext = os.path.splitext(filename)[1]
    if file_ext == ".txt":
        with g_pathmgr.open(filename, "r") as f_open:
            data = f_open.readlines()
    elif file_ext in [".pkl", ".pickle"]:
        with g_pathmgr.open(filename, "rb") as f_open:
            data = pickle.load(f_open, encoding="latin1")
    elif file_ext == ".npy":
        if mmap_mode:
            try:
                with g_pathmgr.open(filename, "rb") as f_open:
                    data = np.load(
                        f_open,
                        allow_pickle=allow_pickle,
                        encoding="latin1",
                        mmap_mode=mmap_mode,
                    )
            except ValueError as e:
                logging.info(
                    f"Could not mmap {filename}: {e}. Trying without g_pathmgr"
                )
                data = np.load(
                    filename,
                    allow_pickle=allow_pickle,
                    encoding="latin1",
                    mmap_mode=mmap_mode,
                )
                logging.info("Successfully loaded without g_pathmgr")
            except Exception:
                logging.info("Could not mmap without g_pathmgr. Trying without mmap")
                with g_pathmgr.open(filename, "rb") as f_open:
                    data = np.load(f_open, allow_pickle=allow_pickle, encoding="latin1")
        else:
            with g_pathmgr.open(filename, "rb") as f_open:
                data = np.load(f_open, allow_pickle=allow_pickle, encoding="latin1")
    elif file_ext == ".json":
        with g_pathmgr.open(filename, "r") as f_open:
            data = json.load(f_open)
    elif file_ext == ".yaml":
        with g_pathmgr.open(filename, "r") as f_open:
            data = yaml.load(f_open, Loader=yaml.FullLoader)
    elif file_ext == ".csv":
        with g_pathmgr.open(filename, "r") as f_open:
            data = pd.read_csv(f_open)
    else:
        raise Exception(f"Reading from {file_ext} is not supported yet")
    return data


def abspath(resource_path: str):
    """
    Make a path absolute, but take into account prefixes like
    "http://" or "manifold://"
    """
    regex = re.compile(r"^\w+://")
    if regex.match(resource_path) is None:
        return os.path.abspath(resource_path)
    else:
        return resource_path


def cleanup_dir(dir):
    """
    Utility for deleting a directory. Useful for cleaning the storage space
    that contains various training artifacts like checkpoints, data etc.
    """
    if os.path.exists(dir):
        logging.info(f"Deleting directory: {dir}")
        shutil.rmtree(dir)
    logging.info(f"Deleted contents of directory: {dir}")


def get_file_size(filename):
    """
    Given a file, get the size of file in MB
    """
    size_in_mb = os.path.getsize(filename) / float(1024 ** 2)
    return size_in_mb


logger = logging.getLogger(__name__)


def setup_logger():
    logging.basicConfig(
        level=logging.INFO if dist_utils.is_main_process() else logging.WARN,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


def pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption


def suppress_errors(return_value=False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in function '{func.__name__}': {e}")
                return return_value  # 오류 발생 시 지정된 기본값 반환

        return wrapper

    return decorator


def dummy_collator(batch: List[Dict]) -> dict:
    """
    A dummy collator function to bypass type errors caused by non-tensor elements in a batch.

    This collator avoids the `TypeError: batch must contain tensors, numbers, dicts or lists;
    found <class 'PIL.Image.Image'>` when handling batches containing non-standard data types
    like PIL images. Instead of performing tensor stacking, it simply returns the batch as-is.

    Args:
        batch (`List[Dict]`): A list of dictionaries, where each dictionary corresponds to an
        element in the batch.

    Returns:
        `Dict`: The first input batch is returned unchanged.

    Note:
        For more information on the default PyTorch collator (`default_collate`), you can refer to its
        implementation in the PyTorch repository:
        https://github.com/pytorch/pytorch/blob/main/torch/utils/data/_utils/collate.py
    """
    return batch[0]
