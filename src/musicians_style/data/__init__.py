"""Akwizytor_Danych i Manifest_Zbioru (Wymagania 1.x)."""

from .acquirer import (
    LOCAL_SOURCE,
    MAX_DURATION_S,
    MAX_YOUTUBE_CLIP_S,
    MIN_DURATION_S,
    MIN_RECOMMENDED_FILES,
    YOUTUBE_SOURCE,
    AudioToMidiConverter,
    DatasetAcquirer,
    YoutubeAudioDownloader,
    YoutubeClip,
    YtDlpAudioDownloader,
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
    "YOUTUBE_SOURCE",
    "MAX_DURATION_S",
    "MAX_YOUTUBE_CLIP_S",
    "MIN_DURATION_S",
    "MIN_RECOMMENDED_FILES",
    "AudioToMidiConverter",
    "YoutubeAudioDownloader",
    "YoutubeClip",
    "YtDlpAudioDownloader",
    "FileEntry",
    "Manifest",
    "compute_sha256",
    "current_git_commit",
    "from_json",
    "to_json",
]
