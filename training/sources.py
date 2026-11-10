# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Records, fetches, and checks the source files by their SHA-256.

SHA256SUMS lists every source file the training data is built from, in the
format of sha256sum. The JMA files are fetched from the JMA website; the
ground data files are downloaded by the user, who records them here. A file
whose hash differs from the record stops the build, so a replaced source is
noticed instead of silently changing the data.

Usage:
    python -m training.sources [--sums PATH] fetch [--directory DIR]
        [--record NAME ...]
    python -m training.sources [--sums PATH] record PATH ...
    python -m training.sources [--sums PATH] verify PATH ...
"""

import argparse
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
import contextlib
import hashlib
import http.client
import logging
import os
import pathlib
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request

from training import features

SHA256SUMS = features.REPOSITORY / "training" / "SHA256SUMS"
JMA_URL = "https://www.data.jma.go.jp/eqev/data/bulletin/data/shindo/"
DEFAULT_JMA_DIRECTORY = features.REPOSITORY / "training" / "data" / "jma"

_NAME = re.compile(r"[^\s/\\]+")
_LINE = re.compile(rf"([0-9a-f]{{64}})  ({_NAME.pattern})")
_JMA_NAME = re.compile(r"code_p\.zip|i[0-9]{4}\.zip")
_BLOCK_BYTES = 1 << 20
_TIMEOUT_SECONDS = 60

_logger = logging.getLogger(__name__)


class SourceError(Exception):
    """A source or SHA256SUMS fails a check, or a source cannot be fetched."""


def read_sums(path: pathlib.Path = SHA256SUMS) -> dict[str, str]:
    """Reads the recorded hashes.

    Args:
        path: The SHA256SUMS file. A missing file records nothing.

    Returns:
        The lower-case hexadecimal SHA-256 of each file name.

    Raises:
        SourceError: The file is not ASCII, a line is not a hash, two spaces,
            and a file name, or a name is recorded twice.
    """
    if not path.exists():
        return {}
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except UnicodeDecodeError as error:
        raise SourceError(f"{path}: not ASCII text") from error
    sums: dict[str, str] = {}
    for number, line in enumerate(lines, start=1):
        match = _LINE.fullmatch(line)
        if match is None:
            raise SourceError(f"{path}:{number}: not '<sha256>  <name>'")
        digest, name = match.groups()
        if name in sums:
            raise SourceError(f"{path}:{number}: {name} is recorded twice")
        sums[name] = digest
    return sums


def verify(path: pathlib.Path, sums: Mapping[str, str]) -> None:
    """Checks a file against its recorded hash.

    Args:
        path: The file. Its name is looked up in sums.
        sums: The recorded hashes from read_sums.

    Raises:
        SourceError: The name is not recorded, or the hash differs.
        OSError: The file cannot be read.
    """
    expected = sums.get(path.name)
    if expected is None:
        raise SourceError(
            f"{path.name} is not recorded in SHA256SUMS; record it with"
            " `python -m training.sources record` after checking its source"
        )
    _check(path, expected)


def record(
    paths: Iterable[pathlib.Path], sums_path: pathlib.Path = SHA256SUMS
) -> None:
    """Adds the hashes of new files to SHA256SUMS.

    The lines are written in the order of the names, with LF line endings,
    to a new temporary file that then replaces SHA256SUMS, so a failed write
    leaves the old records.

    Args:
        paths: The files to record. Their names must not be recorded yet.
        sums_path: The SHA256SUMS file.

    Raises:
        SourceError: A name is recorded already, so recording it would hide
            a replaced source, or a name is not ASCII or holds a space or a
            backslash, so it could not be read back. Nothing is written then.
        OSError: A file cannot be read, or SHA256SUMS cannot be written.
    """
    sums = read_sums(sums_path)
    added: dict[str, str] = {}
    for path in paths:
        if not path.name.isascii() or not _NAME.fullmatch(path.name):
            raise SourceError(
                f"{path.name!r} cannot be recorded; a name must be ASCII"
                " without spaces or backslashes"
            )
        if path.name in sums or path.name in added:
            raise SourceError(
                f"{path.name} is recorded already; check the file with"
                " `python -m training.sources verify` instead"
            )
        added[path.name] = _file_sha256(path)
    sums.update(added)
    text = "".join(f"{sums[name]}  {name}\n" for name in sorted(sums))
    with _temporary_file(sums_path.parent, text.encode("ascii")) as temporary:
        os.replace(temporary, sums_path)


def fetch(
    directory: pathlib.Path,
    *,
    names: Iterable[str] | None = None,
    record_new: Sequence[str] = (),
    sums_path: pathlib.Path = SHA256SUMS,
    opener: Callable[[str], bytes] | None = None,
    delay: float = 1.0,
) -> list[pathlib.Path]:
    """Fetches the JMA files into a directory and checks them.

    A file already in the directory is only checked. A missing file is
    downloaded to a new temporary file, which takes the file's name once its
    hash matches and <name>.part otherwise, replacing a .part left by an
    earlier run without writing through it. The names are checked before
    anything is downloaded.

    Args:
        directory: Where the files go. It is created when missing.
        names: The recorded JMA files to fetch. None fetches every recorded
            code_p.zip and i<year>.zip.
        record_new: Names to download and record, for a new year. They must
            not be recorded yet, appear once, and not be in the directory.
        sums_path: The SHA256SUMS file.
        opener: Returns the bytes at a URL. None uses urllib.
        delay: Seconds to wait after each download.

    Returns:
        The paths of the fetched files, the recorded ones first.

    Raises:
        SourceError: A name is not code_p.zip or i<year>.zip, a name is not
            recorded, a name in record_new is recorded, repeated, or present,
            a hash differs, or a download fails.
        OSError: A file cannot be read or written.
    """
    sums = read_sums(sums_path)
    if names is None:
        selected = sorted(name for name in sums if _JMA_NAME.fullmatch(name))
    else:
        selected = list(names)
    _check_names(directory, selected, record_new, sums)
    read = _read_url if opener is None else opener
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in selected:
        path = directory / name
        if path.exists():
            _check(path, sums[name])
        else:
            _store(path, _download(name, read, delay), sums[name])
        paths.append(path)
    new_paths = []
    for name in record_new:
        path = directory / name
        _store(path, _download(name, read, delay), None)
        new_paths.append(path)
    if new_paths:
        record(new_paths, sums_path)
    return paths + new_paths


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the command line.

    Args:
        argv: The arguments after the program name. None reads sys.argv.

    Returns:
        The exit status: 0 on success, 1 when a source or SHA256SUMS fails a
        check, or a file cannot be read, written, or fetched.
    """
    parser = argparse.ArgumentParser(prog="python -m training.sources")
    parser.add_argument(
        "--sums", type=pathlib.Path, default=SHA256SUMS, help="SHA256SUMS"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fetch_parser = commands.add_parser("fetch", help="fetch the JMA files")
    fetch_parser.add_argument(
        "--directory", type=pathlib.Path, default=DEFAULT_JMA_DIRECTORY
    )
    fetch_parser.add_argument("--record", nargs="+", default=[])
    record_parser = commands.add_parser("record", help="record new files")
    record_parser.add_argument("paths", nargs="+", type=pathlib.Path)
    verify_parser = commands.add_parser("verify", help="check files")
    verify_parser.add_argument("paths", nargs="+", type=pathlib.Path)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if arguments.command == "fetch":
            fetch(
                arguments.directory,
                record_new=arguments.record,
                sums_path=arguments.sums,
            )
        elif arguments.command == "record":
            record(arguments.paths, arguments.sums)
        else:
            sums = read_sums(arguments.sums)
            for path in arguments.paths:
                verify(path, sums)
    except (SourceError, OSError) as error:
        _logger.error("%s", error)
        return 1
    return 0


def _check_names(
    directory: pathlib.Path,
    selected: Sequence[str],
    record_new: Sequence[str],
    sums: Mapping[str, str],
) -> None:
    for name in [*selected, *record_new]:
        if not _JMA_NAME.fullmatch(name):
            raise SourceError(
                f"{name!r} is not a JMA file name (code_p.zip or i<year>.zip)"
            )
    for name in selected:
        if name not in sums:
            raise SourceError(
                f"{name} is not recorded in SHA256SUMS; record it with"
                " `python -m training.sources fetch --record`"
            )
    if len(set(record_new)) != len(record_new):
        raise SourceError(f"--record repeats a name: {" ".join(record_new)}")
    for name in record_new:
        if name in sums:
            raise SourceError(
                f"{name} is recorded already; fetch it without --record"
            )
        if (directory / name).exists():
            raise SourceError(
                f"{directory / name} is present already; check it and record"
                " it with `python -m training.sources record`"
            )


def _download(name: str, read: Callable[[str], bytes], delay: float) -> bytes:
    _logger.info("fetching %s", name)
    data = read(JMA_URL + name)
    time.sleep(delay)
    return data


def _store(path: pathlib.Path, data: bytes, expected: str | None) -> None:
    """Puts downloaded bytes at path, or at <name>.part if the hash differs.

    Args:
        path: Where the bytes go.
        data: The downloaded bytes.
        expected: The recorded SHA-256, or None for a file being recorded.

    Raises:
        SourceError: The hash differs from expected. The bytes are left at
            <name>.part for inspection.
    """
    part = path.with_name(f"{path.name}.part")
    actual = hashlib.sha256(data).hexdigest()
    if expected is not None and actual != expected:
        _put(part, data)
        raise SourceError(_mismatch(part, actual, expected))
    _put(path, data)
    part.unlink(missing_ok=True)


def _put(path: pathlib.Path, data: bytes) -> None:
    """Writes a file by renaming a new one onto it, so a link is replaced."""
    with _temporary_file(path.parent, data) as temporary:
        os.replace(temporary, path)


@contextlib.contextmanager
def _temporary_file(
    directory: pathlib.Path, data: bytes
) -> Iterator[pathlib.Path]:
    """Creates a file in directory that holds data, for renaming it elsewhere.

    Returns:
        A context manager providing the path of the new file, which is
        removed on exit unless it has been renamed.
    """
    descriptor, name = tempfile.mkstemp(
        dir=directory, prefix=".", suffix=".tmp"
    )
    path = pathlib.Path(name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(data)
        yield path
    finally:
        path.unlink(missing_ok=True)


def _check(path: pathlib.Path, expected: str) -> None:
    actual = _file_sha256(path)
    if actual != expected:
        raise SourceError(_mismatch(path, actual, expected))


def _mismatch(path: pathlib.Path, actual: str, expected: str) -> str:
    return (
        f"{path}: SHA-256 {actual} differs from {expected} in SHA256SUMS;"
        " the source may have been replaced"
    )


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _read_url(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_SECONDS) as response:
            data: bytes = response.read()
    except (urllib.error.URLError, http.client.HTTPException) as error:
        raise SourceError(f"{url}: {error}") from error
    return data


if __name__ == "__main__":
    sys.exit(main())
