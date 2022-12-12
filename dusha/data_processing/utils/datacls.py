from dataclasses import dataclass
from typing import Any

# -----------------------------------------------------------------------------
#                               AGGREGATION
# -----------------------------------------------------------------------------


@dataclass
class DawidSkeneEntryDataclass:
    task: str
    worker: str
    label: Any


@dataclass
class DawidSkeneResultDataclass:
    task: str
    pred: str


@dataclass
class MarkupDataclass:
    hash_id: str
    audio_path: str
    duration: str
    annotator_emo: str
    golden_emo: str
    speaker_text: str
    speaker_emo: str
    source_id: str
    audio_path: str
    annotator_emo: str
    annotator_id: str


@dataclass
class AggDataclass:
    hash_id: str
    audio_path: str
    duration: str
    emotion: str
    golden_emo: str
    speaker_text: str
    speaker_emo: str
    source_id: str


# -----------------------------------------------------------------------------
#                               FEATURES
# -----------------------------------------------------------------------------


@dataclass
class DataWithFeaturesEntryclass:
    wav_path: str
    wav_id: str


# -----------------------------------------------------------------------------
#                               EXP
# -----------------------------------------------------------------------------


@dataclass
class DataForExp:
    id: str
    tensor: str
    wav_length: str
    label: int
    emotion: str
