import hashlib
import os
import shutil
import zipfile
import tarfile
import urllib
import requests
from tqdm import tqdm

# ETH proxy (required on Euler)
PROXIES = {
    "http": "http://proxy.ethz.ch:3128",
    "https": "http://proxy.ethz.ch:3128"
}

def _download(url, fname, chunk_size=1024):
    """Downloads file using requests and shows a progress bar."""
    with requests.get(url, stream=True, proxies=PROXIES) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get('content-length', 0))
        with open(fname, 'wb') as file, tqdm(
            desc=fname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)


def extract_archive(file_path, path='.', archive_format='auto'):
    """Extracts .tar, .zip etc."""
    if archive_format == 'auto':
        archive_format = ['tar', 'zip']
    elif isinstance(archive_format, str):
        archive_format = [archive_format]

    for archive_type in archive_format:
        if archive_type == 'tar':
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        elif archive_type == 'zip':
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile
        else:
            continue

        if is_match_fn(file_path):
            with open_fn(file_path) as archive:
                try:
                    archive.extractall(path)
                except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
                    if os.path.exists(path):
                        shutil.rmtree(path)
                    raise
            return True
    return False


def _hash_file(fpath, algorithm='md5', chunk_size=131072):
    hasher = hashlib.new(algorithm)
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_file(fpath, file_hash, algorithm='md5', chunk_size=131072):
    return _hash_file(fpath, algorithm, chunk_size) == file_hash


def get_file(origin=None,
             fname=None,
             file_hash=None,
             datadir='datasets',
             hash_algorithm='md5',
             extract=False,
             force_download=False,
             archive_format='auto'):

    if origin is None:
        raise ValueError("You must specify 'origin' (a download URL).")

    os.makedirs(datadir, exist_ok=True)

    if not fname:
        fname = os.path.basename(urllib.parse.urlsplit(origin).path)
    fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath) and not force_download:
        if file_hash and not validate_file(fpath, file_hash, algorithm=hash_algorithm):
            print("⚠️ Hash mismatch. Re-downloading...")
            download = True
        else:
            print("✅ File already exists and is valid.")
    else:
        download = True

    if download:
        print(f"📥 Downloading from {origin} to {fpath}")
        try:
            _download(origin, fpath)
        except requests.exceptions.RequestException as e:
            raise Exception(f"Download failed for {origin}: {e}")

        if file_hash and not validate_file(fpath, file_hash, algorithm=hash_algorithm):
            os.remove(fpath)
            raise RuntimeError(f"❌ Checksum mismatch after download: {fpath}")

    if extract:
        print(f"📦 Extracting {fpath}")
        extract_archive(fpath, datadir, archive_format)

    return fpath, download
