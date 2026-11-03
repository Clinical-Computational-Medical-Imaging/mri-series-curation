"""Where the DICOM files come from. A source is a set of studies; a study is a set of member files.

  ZipSource     one ZIP file per study (e.g. a PACS / Orthanc export)
  FolderSource  one sub-folder per study (any depth below it), e.g. a DICOM export on disk

Only the grouping into studies differs; series are always found by SeriesInstanceUID inside a
study, so it does not matter how files are named or nested. Index entries keep (study, member) and
read the files through the same source again when volumes are loaded.
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path


class ZipSource:
    kind = "zip"

    def __init__(self, root: str):
        self.root = Path(root)

    def studies(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.suffix.lower() == ".zip")

    def members(self, study: str) -> list[str]:
        with zipfile.ZipFile(self.root / study) as z:
            return [n for n in z.namelist() if not n.endswith("/")]

    def reader(self, study: str) -> "_ZipReader":
        """Context manager with .open(member) -> binary file object; keeps the ZIP open."""
        return _ZipReader(self.root / study)


class _ZipReader:
    def __init__(self, path: Path):
        self._z = zipfile.ZipFile(path)

    def open(self, member: str):
        return self._z.open(member)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._z.close()


class FolderSource:
    kind = "folder"

    def __init__(self, root: str):
        self.root = Path(root)

    def studies(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def members(self, study: str) -> list[str]:
        base = self.root / study
        return sorted(str(p.relative_to(base)) for p in base.rglob("*") if p.is_file())

    def reader(self, study: str) -> "_FolderReader":
        return _FolderReader(self.root / study)


class _FolderReader:
    def __init__(self, base: Path):
        self._base = base

    def open(self, member: str):
        return io.open(os.path.join(self._base, member), "rb")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


def make_source(kind: str, root: str):
    if kind == "zip":
        return ZipSource(root)
    if kind == "folder":
        return FolderSource(root)
    raise ValueError(f"input.kind must be 'zip' or 'folder', got {kind!r}")
