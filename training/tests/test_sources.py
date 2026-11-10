# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Tests that sources are recorded, fetched, and checked by their SHA-256."""

import hashlib
import http.client
import os
import pathlib
import tempfile
import unittest
from unittest import mock
import urllib.error

from training import sources

_FILES = {
    "code_p.zip": b"stations",
    "i2004.zip": b"bulletin 2004",
    "i2025.zip": b"bulletin 2025",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_SUMS = {name: _sha256(data) for name, data in _FILES.items()}


class _FailingFile:
    """Wraps a file to fail halfway through a write or when it closes."""

    def __init__(self, file, failing):
        self._file = file
        self._failing = failing

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()

    def write(self, data):
        if self._failing == "write":
            self._file.write(data[: len(data) // 2])
            raise OSError("no space left on device")
        return self._file.write(data)

    def close(self):
        self._file.close()
        if self._failing == "close":
            raise OSError("no space left on device")


def _failing_fdopen(failing):
    """Returns an os.fdopen whose files fail like _FailingFile."""
    fdopen = os.fdopen

    def failing_fdopen(descriptor, mode):
        return _FailingFile(fdopen(descriptor, mode), failing)

    return failing_fdopen


class SourcesTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.root = pathlib.Path(
            self.enterContext(tempfile.TemporaryDirectory())
        )
        self.sums = self.root / "SHA256SUMS"
        self.directory = self.root / "jma"
        self.urls = []

    def _opener(self, url):
        self.urls.append(url)
        return _FILES[url.removeprefix(sources.JMA_URL)]

    def _write_sums(self, names):
        self.sums.write_text(
            "".join(f"{_SUMS[name]}  {name}\n" for name in names),
            encoding="ascii",
            newline="\n",
        )

    def _fetch(self, **options):
        return sources.fetch(
            self.directory,
            sums_path=self.sums,
            opener=self._opener,
            delay=0.0,
            **options,
        )

    def test_reads_and_writes_sums(self):
        paths = []
        for name in ("i2004.zip", "code_p.zip"):
            path = self.root / name
            path.write_bytes(_FILES[name])
            paths.append(path)
        sources.record(paths, self.sums)
        self.assertEqual(
            self.sums.read_bytes(),
            (
                f"{_SUMS["code_p.zip"]}  code_p.zip\n"
                f"{_SUMS["i2004.zip"]}  i2004.zip\n"
            ).encode("ascii"),
        )
        self.assertEqual(
            sources.read_sums(self.sums),
            {
                "code_p.zip": _sha256(_FILES["code_p.zip"]),
                "i2004.zip": _sha256(_FILES["i2004.zip"]),
            },
        )

    def test_missing_sums_read_as_empty(self):
        self.assertEqual(sources.read_sums(self.sums), {})

    def test_malformed_lines_are_refused(self):
        digest = _sha256(b"x")
        lines = {
            "one space": f"{digest} a.zip\n",
            "63 digits": f"{digest[:-1]}  a.zip\n",
            "upper case": f"{digest.upper()}  a.zip\n",
            "duplicate name": f"{digest}  a.zip\n{digest}  a.zip\n",
        }
        for case, text in lines.items():
            with self.subTest(case=case):
                self.sums.write_text(text, encoding="ascii", newline="\n")
                with self.assertRaises(sources.SourceError):
                    sources.read_sums(self.sums)

    def test_verify_accepts_the_recorded_file(self):
        self._write_sums(["code_p.zip"])
        path = self.root / "code_p.zip"
        path.write_bytes(_FILES["code_p.zip"])
        sources.verify(path, sources.read_sums(self.sums))

    def test_verify_names_both_hashes_on_a_mismatch(self):
        self._write_sums(["code_p.zip"])
        path = self.root / "code_p.zip"
        path.write_bytes(b"replaced")
        with self.assertRaises(sources.SourceError) as context:
            sources.verify(path, sources.read_sums(self.sums))
        message = str(context.exception)
        self.assertIn(str(path), message)
        self.assertIn(_sha256(b"replaced"), message)
        self.assertIn(_sha256(_FILES["code_p.zip"]), message)

    def test_unrecorded_names_are_refused(self):
        self._write_sums(["code_p.zip"])
        path = self.root / "i2004.zip"
        path.write_bytes(_FILES["i2004.zip"])
        with self.assertRaisesRegex(sources.SourceError, "record"):
            sources.verify(path, sources.read_sums(self.sums))
        with self.assertRaisesRegex(sources.SourceError, "record"):
            self._fetch(names=["i2030.zip"])
        self.assertEqual(self.urls, [])

    def test_fetch_downloads_and_checks_recorded_files(self):
        self._write_sums(["code_p.zip", "i2004.zip"])
        paths = self._fetch()
        self.assertEqual(
            paths,
            [self.directory / "code_p.zip", self.directory / "i2004.zip"],
        )
        for path in paths:
            self.assertEqual(path.read_bytes(), _FILES[path.name])
        self.assertEqual(list(self.directory.glob("*.part")), [])
        self.assertEqual(
            self.urls,
            [sources.JMA_URL + "code_p.zip", sources.JMA_URL + "i2004.zip"],
        )

    def test_fetch_only_takes_bulletin_files(self):
        self.sums.write_text(
            f"{_sha256(b"ground")}  ground.zip\n"
            f"{_SUMS["i2004.zip"]}  i2004.zip\n",
            encoding="ascii",
            newline="\n",
        )
        self.assertEqual(self._fetch(), [self.directory / "i2004.zip"])

    def test_fetch_can_take_some_names(self):
        self._write_sums(["code_p.zip", "i2004.zip"])
        self.assertEqual(
            self._fetch(names=["i2004.zip"]), [self.directory / "i2004.zip"]
        )
        self.assertEqual(self.urls, [sources.JMA_URL + "i2004.zip"])

    def test_fetch_skips_files_already_present(self):
        self._write_sums(["code_p.zip"])
        self._fetch()
        self.urls.clear()
        self.assertEqual(self._fetch(), [self.directory / "code_p.zip"])
        self.assertEqual(self.urls, [])

    def test_mismatch_stops_without_replacing_the_file(self):
        self._write_sums(["code_p.zip"])
        self.directory.mkdir()
        present = self.directory / "code_p.zip"
        present.write_bytes(b"replaced")
        with self.assertRaises(sources.SourceError):
            self._fetch()
        self.assertEqual(present.read_bytes(), b"replaced")
        self.assertEqual(self.urls, [])

        present.unlink()
        self.sums.write_text(
            f"{_sha256(b"other")}  code_p.zip\n",
            encoding="ascii",
            newline="\n",
        )
        with self.assertRaises(sources.SourceError):
            self._fetch()
        self.assertFalse(present.exists())
        self.assertEqual(
            (self.directory / "code_p.zip.part").read_bytes(),
            _FILES["code_p.zip"],
        )

    def test_record_new_adds_fetched_files(self):
        self._write_sums(["code_p.zip"])
        paths = self._fetch(names=[], record_new=["i2025.zip"])
        self.assertEqual(paths, [self.directory / "i2025.zip"])
        self.assertEqual(
            sources.read_sums(self.sums)["i2025.zip"],
            _sha256(_FILES["i2025.zip"]),
        )

        before = self.sums.read_bytes()
        self.urls.clear()
        with self.assertRaises(sources.SourceError):
            self._fetch(names=[], record_new=["code_p.zip"])
        self.assertEqual(self.sums.read_bytes(), before)
        self.assertEqual(self.urls, [])

    def test_record_refuses_recorded_names(self):
        self._write_sums(["code_p.zip"])
        before = self.sums.read_bytes()
        paths = []
        for name, data in (("i2004.zip", b"a"), ("code_p.zip", b"b")):
            path = self.root / name
            path.write_bytes(data)
            paths.append(path)
        with self.assertRaisesRegex(sources.SourceError, "code_p.zip"):
            sources.record(paths, self.sums)
        self.assertEqual(self.sums.read_bytes(), before)

    def test_record_keeps_sums_when_writing_fails(self):
        self._write_sums(["code_p.zip"])
        before = self.sums.read_bytes()
        path = self.root / "i2004.zip"
        path.write_bytes(_FILES["i2004.zip"])
        with mock.patch.object(
            sources.os, "replace", side_effect=OSError("disk full")
        ):
            with self.assertRaises(OSError):
                sources.record([path], self.sums)
        self.assertEqual(self.sums.read_bytes(), before)
        self.assertEqual(
            sorted(child.name for child in self.root.iterdir()),
            ["SHA256SUMS", "i2004.zip"],
        )

    def test_record_refuses_names_it_cannot_read_back(self):
        self._write_sums(["code_p.zip"])
        before = self.sums.read_bytes()
        for name in ("a b.zip", "a\\b.zip", chr(0xE9) + ".zip"):
            with self.subTest(name=name):
                path = self.root / name
                path.write_bytes(b"x")
                with self.assertRaises(sources.SourceError):
                    sources.record([path], self.sums)
                self.assertEqual(self.sums.read_bytes(), before)

    def test_fetch_refuses_names_that_are_not_jma_files(self):
        self._write_sums(["code_p.zip"])
        names = (
            "../i2025.zip",
            "/tmp/i2025.zip",
            "sub/i2025.zip",
            "ground.zip",
        )
        for name in names:
            with self.subTest(name=name):
                with self.assertRaises(sources.SourceError):
                    self._fetch(names=[], record_new=[name])
                with self.assertRaises(sources.SourceError):
                    self._fetch(names=[name])
        self.assertEqual(self.urls, [])

    def test_record_new_refuses_files_already_present(self):
        self._write_sums(["code_p.zip"])
        self.directory.mkdir()
        present = self.directory / "i2025.zip"
        present.write_bytes(b"kept")
        with self.assertRaisesRegex(sources.SourceError, "record"):
            self._fetch(names=[], record_new=["i2025.zip"])
        self.assertEqual(present.read_bytes(), b"kept")
        self.assertEqual(self.urls, [])

    def test_record_new_refuses_duplicate_names(self):
        self._write_sums(["code_p.zip"])
        with self.assertRaises(sources.SourceError):
            self._fetch(names=[], record_new=["i2025.zip", "i2025.zip"])
        self.assertEqual(self.urls, [])

    def test_fetch_waits_after_each_download(self):
        self._write_sums(["code_p.zip", "i2004.zip"])
        self.directory.mkdir()
        (self.directory / "code_p.zip").write_bytes(_FILES["code_p.zip"])
        events = []

        def opener(url):
            name = url.removeprefix(sources.JMA_URL)
            events.append(name)
            return _FILES[name]

        with mock.patch.object(
            sources.time, "sleep", side_effect=events.append
        ):
            sources.fetch(
                self.directory,
                record_new=["i2025.zip"],
                sums_path=self.sums,
                opener=opener,
                delay=1.5,
            )
        self.assertEqual(events, ["i2004.zip", 1.5, "i2025.zip", 1.5])

    def test_failed_download_keeps_earlier_files(self):
        self._write_sums(["code_p.zip"])
        before = self.sums.read_bytes()

        def opener(url):
            name = url.removeprefix(sources.JMA_URL)
            if name == "i2025.zip":
                raise sources.SourceError(f"{url}: connection reset")
            return _FILES[name]

        with self.assertRaisesRegex(sources.SourceError, "connection reset"):
            sources.fetch(
                self.directory,
                record_new=["i2025.zip"],
                sums_path=self.sums,
                opener=opener,
                delay=0.0,
            )
        self.assertEqual(
            (self.directory / "code_p.zip").read_bytes(), _FILES["code_p.zip"]
        )
        self.assertFalse((self.directory / "i2025.zip").exists())
        self.assertEqual(self.sums.read_bytes(), before)

    def test_download_errors_become_source_errors(self):
        self._write_sums(["code_p.zip"])
        errors = {
            "URLError": urllib.error.URLError("unreachable"),
            "IncompleteRead": http.client.IncompleteRead(b"part"),
        }
        url = sources.JMA_URL + "code_p.zip"
        for case, error in errors.items():
            with (
                self.subTest(case=case),
                mock.patch.object(
                    sources.urllib.request, "urlopen", side_effect=error
                ),
                self.assertRaisesRegex(sources.SourceError, url),
            ):
                sources.fetch(self.directory, sums_path=self.sums, delay=0.0)

    def test_non_ascii_sums_are_refused(self):
        self.sums.write_bytes(b"\xff\n")
        with self.assertRaisesRegex(sources.SourceError, "ASCII"):
            sources.read_sums(self.sums)
        path = self.root / "code_p.zip"
        path.write_bytes(_FILES["code_p.zip"])
        with self.assertLogs(level="ERROR"):
            status = sources.main(
                ["--sums", str(self.sums), "verify", str(path)]
            )
        self.assertEqual(status, 1)

    def test_record_leaves_other_temporary_files(self):
        other = self.root / "SHA256SUMS.tmp"
        other.write_bytes(b"other")
        sources.record([other], self.sums)
        self.assertEqual(other.read_bytes(), b"other")
        sources.verify(other, sources.read_sums(self.sums))

    def test_failed_writes_leave_no_files(self):
        self._write_sums(["code_p.zip"])
        before = self.sums.read_bytes()
        path = self.root / "i2004.zip"
        path.write_bytes(_FILES["i2004.zip"])
        for failing in ("write", "close"):
            with self.subTest(failing=failing):
                with mock.patch.object(
                    sources.os, "fdopen", _failing_fdopen(failing)
                ):
                    with self.assertRaises(OSError):
                        sources.record([path], self.sums)
                    with self.assertRaises(OSError):
                        self._fetch()
                self.assertEqual(self.sums.read_bytes(), before)
                self.assertEqual(
                    sorted(child.name for child in self.root.iterdir()),
                    ["SHA256SUMS", "i2004.zip", "jma"],
                )
                self.assertEqual(list(self.directory.iterdir()), [])

    def test_failed_rename_leaves_no_files(self):
        self._write_sums(["code_p.zip"])
        with (
            mock.patch.object(
                sources.os, "replace", side_effect=OSError("read-only")
            ),
            self.assertRaises(OSError),
        ):
            self._fetch()
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_stale_part_is_replaced(self):
        self._write_sums(["code_p.zip"])
        self.directory.mkdir()
        part = self.directory / "code_p.zip.part"
        part.write_bytes(b"stale")
        self._fetch()
        self.assertEqual(
            sorted(child.name for child in self.directory.iterdir()),
            ["code_p.zip"],
        )
        self.assertEqual(
            (self.directory / "code_p.zip").read_bytes(), _FILES["code_p.zip"]
        )

    def test_mismatched_part_does_not_write_through_a_link(self):
        self.sums.write_text(
            f"{_sha256(b"other")}  code_p.zip\n",
            encoding="ascii",
            newline="\n",
        )
        self.directory.mkdir()
        target = self.root / "linked"
        target.write_bytes(b"kept")
        (self.directory / "code_p.zip.part").symlink_to(target)
        with self.assertRaises(sources.SourceError):
            self._fetch()
        self.assertEqual(target.read_bytes(), b"kept")
        self.assertEqual(
            (self.directory / "code_p.zip.part").read_bytes(),
            _FILES["code_p.zip"],
        )

    def test_retry_after_a_failed_new_download_keeps_the_first(self):
        self._write_sums(["code_p.zip"])

        def opener(url):
            self.urls.append(url)
            name = url.removeprefix(sources.JMA_URL)
            if name == "i2026.zip":
                raise sources.SourceError(f"{url}: connection reset")
            return _FILES[name]

        options = {
            "names": [],
            "record_new": ["i2025.zip", "i2026.zip"],
            "sums_path": self.sums,
            "opener": opener,
            "delay": 0.0,
        }
        with self.assertRaises(sources.SourceError):
            sources.fetch(self.directory, **options)
        self.urls.clear()
        with self.assertRaisesRegex(sources.SourceError, "present already"):
            sources.fetch(self.directory, **options)
        self.assertEqual(self.urls, [])
        self.assertNotIn("i2025.zip", sources.read_sums(self.sums))

    def test_cli_reports_errors(self):
        self._write_sums(["code_p.zip"])
        path = self.root / "code_p.zip"
        path.write_bytes(b"replaced")
        with self.assertLogs(level="ERROR") as logs:
            status = sources.main(
                ["--sums", str(self.sums), "verify", str(path)]
            )
        self.assertEqual(status, 1)
        self.assertIn("SHA-256", logs.output[0])


if __name__ == "__main__":
    unittest.main()
