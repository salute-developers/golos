from pathlib import Path
from typing import List, Set

import librosa
import numpy as np
from tqdm import tqdm
from utils.datacls import DataWithFeaturesEntryclass


def create_features(
    data: List[DataWithFeaturesEntryclass],
    wavs_names: Set[str],
    features_dump_path: Path,
    dataset_name: str,
    recalculate_feature: bool,
    hop_length_coef: float = 0.01,
    win_length_coef: float = 0.02,
    sample_rate: int = 16000,
    n_mels: int = 64,
) -> None:
    """
    As an input all models use standard speech features:
    64 Mel-filterbank calculated from 20ms windows with a 10ms overlap.
    """
    if recalculate_feature:
        if len(data) != len(wavs_names):
            print(
                f"{len(wavs_names) - len(data)} wav files are missing for {dataset_name}"
            )
        hop_length = int(sample_rate * hop_length_coef)
        win_length = int(sample_rate * win_length_coef)
        for row in tqdm(data):
            data, rate = librosa.load(row.wav_path, sr=sample_rate)
            if len(data) != 0:
                spec = librosa.feature.melspectrogram(
                    y=data,
                    sr=rate,
                    hop_length=hop_length,
                    n_fft=win_length,
                    n_mels=n_mels,
                )
            else:
                raise AttributeError
            mel_spec = librosa.power_to_db(spec, ref=np.max)
            np.save(features_dump_path / f"{row.wav_id}.npy", mel_spec[None])
        print(
            f"({len(data)}/{len(wavs_names)}) features have been calculated for {dataset_name}"
        )
    else:
        ready_features = {elm.stem for elm in features_dump_path.glob("*.npy")}
        wav_to_features = {wav for wav in wavs_names if not wav in ready_features}
        data_to_culc = [wav for wav in data if wav.wav_id in wav_to_features]

        if len(data_to_culc) != len(wav_to_features):
            print(
                f"{len(wav_to_features) - len(data_to_culc)} wav files are missing for {dataset_name}"
            )

        if not data_to_culc:
            print(
                f"All({len({wav for wav in wavs_names if wav in ready_features})}/{len(wavs_names)}) features have been calculated for {dataset_name}"
            )
            return

        hop_length = int(sample_rate * hop_length_coef)
        win_length = int(sample_rate * win_length_coef)
        for row in tqdm(data_to_culc):
            data, rate = librosa.load(row.wav_path, sr=sample_rate)
            if len(data) != 0:
                spec = librosa.feature.melspectrogram(
                    y=data,
                    sr=rate,
                    hop_length=hop_length,
                    n_fft=win_length,
                    n_mels=n_mels,
                )
            else:
                raise AttributeError
            mel_spec = librosa.power_to_db(spec, ref=np.max)
            np.save(features_dump_path / f"{row.wav_id}.npy", mel_spec[None])


def load_features(
    wavs_path: Path,
    wavs_names: Set[str],
    result_dir: Path,
    dataset_name: str,
    recalculate_feature: bool,
) -> None:
    wavs = []
    for elm in wavs_path.glob("*.wav"):
        wavs.append(DataWithFeaturesEntryclass(wav_path=str(elm), wav_id=elm.stem))
    create_features(
        data=wavs,
        wavs_names=wavs_names,
        features_dump_path=result_dir / "features",
        dataset_name=dataset_name,
        recalculate_feature=recalculate_feature,
    )
