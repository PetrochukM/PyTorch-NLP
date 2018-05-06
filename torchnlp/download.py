from urllib.parse import urlparse

import logging
import os
import requests
import tarfile
import urllib.request
import zipfile

from tqdm import tqdm

logger = logging.getLogger(__name__)


def _reporthook(t):
    """ ``reporthook`` to use with ``urllib.request`` that prints the process of the download.

    Uses ``tqdm`` for progress bar.

    **Reference:**
    https://github.com/tqdm/tqdm

    Args:
        t (tqdm.tqdm) Progress bar.

    Example:
        >>> with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
        >>>    urllib.request.urlretrieve(file_url, filename=full_path, reporthook=reporthook(t))

    """
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        Args:
            b (int, optional): Number of blocks just transferred [default: 1].
            bsize (int, optional): Size of each block (in tqdm units) [default: 1].
            tsize (int, optional): Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


def _download_file_from_drive(filename, url):  # pragma: no cover
    """ Download filename from google drive unless it's already in directory.

    Args:
        filename (str): Name of the file to download to (do nothing if it already exists).
        url (str): URL to download from.
    """
    confirm_token = None

    # Since the file is big, drive will scan it for virus and take it to a
    # warning page. We find the confirm token on this page and append it to the
    # URL to start the download process.
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token

    logger.info("Downloading %s to %s" % (url, filename))

    response = session.get(url, stream=True)
    # Now begin the download.
    chunk_size = 16 * 1024
    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)

    # Print newline to clear the carriage return from the download progress
    statinfo = os.stat(filename)
    logger.info("Successfully downloaded %s, %s bytes." % (filename, statinfo.st_size))


def _maybe_extract(compressed_filename, directory, extension=None):
    """ Extract a compressed file to ``directory``.

    Args:
        compressed_filename (str): Compressed file.
        directory (str): Extract to directory.
        extension (str, optional): Extension of the file; Otherwise, attempts to extract extension
            from the filename.
    """
    logger.info('Extracting {}'.format(compressed_filename))

    if extension is None:
        basename = os.path.basename(compressed_filename)
        extension = basename.split('.', 1)[1]

    if 'zip' in extension:
        with zipfile.ZipFile(compressed_filename, "r") as zip_:
            zip_.extractall(directory)
    elif 'tar' in extension or 'tgz' in extension:
        with tarfile.open(compressed_filename, mode='r') as tar:
            tar.extractall(path=directory)

    logger.info('Extracted {}'.format(compressed_filename))


def _get_filename_from_url(url):
    """ Return a filename from a URL

    Args:
        url (str): URL to extract filename from

    Returns:
        (str): Filename in URL
    """
    parse = urlparse(url)
    return os.path.basename(parse.path)


def download_file_maybe_extract(url, directory, filename=None, extension=None, check_files=[]):
    """ Download the file at ``url`` to ``directory``. Extract to ``directory`` if tar or zip.

    Args:
        url (str): Url of file.
        directory (str): Directory to download to.
        filename (str, optional): Name of the file to download; Otherwise, a filename is extracted
            from the url.
        extension (str, optional): Extension of the file; Otherwise, attempts to extract extension
            from the filename.
        check_files (list of str): Check if these files exist, ensuring the download succeeded.
            If these files exist before the download, the download is skipped.

    Returns:
        (str): Filename of download file.

    Raises:
        (ValueError): Error if one of the ``check_files`` are not found following the download.
    """
    if filename is None:
        filename = _get_filename_from_url(url)

    filepath = os.path.join(directory, filename)
    check_files = [os.path.join(directory, f) for f in check_files]

    if len(check_files) > 0 and _check_download(*check_files):
        return filepath

    if not os.path.isdir(directory):
        os.makedirs(directory)

    logger.info('Downloading {}'.format(filename))

    # Download
    if 'drive.google.com' in url:
        _download_file_from_drive(filepath, url)
    else:
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            urllib.request.urlretrieve(url, filename=filepath, reporthook=_reporthook(t))

    _maybe_extract(compressed_filename=filepath, directory=directory, extension=extension)

    if not _check_download(*check_files):
        raise ValueError('[DOWNLOAD FAILED] `*check_files` not found')

    return filepath


def _check_download(*filepaths):
    """ Check if the downloaded files are found.

    Args:
        filepaths (list of str): Check if these filepaths exist

    Returns:
        (bool): Returns True if all filepaths exist
    """
    return all([os.path.isfile(filepath) for filepath in filepaths])


def download_files_maybe_extract(urls, directory, check_files=[]):
    """ Download the files at ``urls`` to ``directory``. Extract to ``directory`` if tar or zip.

    Args:
        urls (str): Url of files.
        directory (str): Directory to download to.
        check_files (list of str): Check if these files exist, ensuring the download succeeded.
            If these files exist before the download, the download is skipped.

    Raises:
        (ValueError): Error if one of the ``check_files`` are not found following the download.
    """
    check_files = [os.path.join(directory, f) for f in check_files]
    if _check_download(*check_files):
        return

    for url in urls:
        download_file_maybe_extract(url=url, directory=directory)

    if not _check_download(*check_files):
        raise ValueError('[DOWNLOAD FAILED] `*check_files` not found')
