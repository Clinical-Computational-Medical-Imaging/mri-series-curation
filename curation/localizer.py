"""Localizer / scout / screenshot detection (config `localizer`): an image group is a localizer if
its ImageType says LOCALIZER, if it has few images AND a small matrix, or if its description
contains one of the keywords. Localizer series are never trained and never labelled."""
from __future__ import annotations

from .dicom_index import group_class


def is_localizer(entry: dict | None, conf: dict) -> bool:
    if not entry:
        return False
    if group_class(entry.get("group", "m")) == "loc":
        return True
    desc = str(entry.get("series_description", "") or "").lower()
    if any(k.lower() in desc for k in conf.get("keywords", [])):
        return True
    n = int(entry.get("n_slices_total", 0) or 0)
    if n > conf["max_slices"]:
        return False
    matrix = max(int(entry.get("rows", 0) or 0), int(entry.get("columns", 0) or 0))
    return matrix <= conf["max_matrix"] if matrix else n <= conf["max_slices_without_matrix"]
