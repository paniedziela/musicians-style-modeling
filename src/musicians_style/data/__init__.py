"""Akwizytor_Danych i Manifest_Zbioru (Wymagania 1.x)."""

from .acquirer import (
    LOCAL_SOURCE,
    MAX_DURATION_S,
    MIN_DURATION_S,
    MIN_RECOMMENDED_FILES,
    DatasetAcquirer,
)
from .manifest import (
    FileEntry,
    Manifest,
    compute_sha256,
    current_git_commit,
    from_json,
    to_json,
)

__all__ = [
    "DatasetAcquirer",
    "LOCAL_SOURCE",
    "MAX_DURATION_S",
    "MIN_DURATION_S",
    "MIN_RECOMMENDED_FILES",
    "FileEntry",
    "Manifest",
    "compute_sha256",
    "current_git_commit",
    "from_json",
    "to_json",
]
