import json
import os
from enum import Enum
from pathlib import Path
from typing import Dict, List

from utils.datacls import (
    AggDataclass,
    DataForExp,
    DawidSkeneEntryDataclass,
    MarkupDataclass,
)
from utils.dawidskene import get_dawidskene_pred

HEADER = "\t".join(
    [
        "hash_id",
        "wav_path",
        "duration",
        "emotion",
        "golden_emo",
        "speaker_text",
        "speaker_emo",
        "source_id",
    ]
)

HEADER_EXP = "\t".join(["id", "tensor", "wav_lengh", "label"])


class Emotion(Enum):
    ANGRY = 0
    SAD = 1
    NEUTRAL = 2
    POSITIVE = 3


def read_data_markup(dataset_path: Path, use_tsv: bool) -> List[MarkupDataclass]:
    markup_data = []
    if use_tsv:
        with open(
            dataset_path.parent / (dataset_path.stem + ".tsv"), "r", encoding="utf-8"
        ) as file:
            headers = file.readline().rstrip("\r\n").split("\t")
            for line in file:
                line_data = line.strip("\r\n").split("\t")
                string = dict(zip(headers, line_data))
                row = MarkupDataclass(**string)
                markup_data.append(row)
    else:
        with open(
            dataset_path.parent / (dataset_path.stem + ".jsonl"), "r", encoding="utf-8"
        ) as file:
            for line in file:
                row = MarkupDataclass(**json.loads(line))
                markup_data.append(row)
    return markup_data


def agg_data_to_file(
    file_path: Path, agg_data: List[AggDataclass], use_tsv: bool
) -> None:
    if use_tsv:
        with open(
            file_path.parent / (file_path.stem + ".tsv"), "w", encoding="utf-8"
        ) as file:
            print(HEADER, file=file, end=os.linesep)
            for row in agg_data:
                print("\t".join(row.__dict__.values()), file=file, end=os.linesep)
    else:
        with open(
            file_path.parent / (file_path.stem + ".jsonl"), "w", encoding="utf-8"
        ) as file:
            for row in agg_data:
                line = json.dumps(row.__dict__, ensure_ascii=False)
                print(line, file=file, end=os.linesep)


def exp_data_to_file(
    file_path: Path, exp_data: List[DataForExp], use_tsv: bool
) -> None:
    if use_tsv:
        with open(
            file_path.parent / (file_path.stem + ".tsv"), "w", encoding="utf-8"
        ) as file:
            print(HEADER_EXP, file=file, end=os.linesep)
            for row in exp_data:
                line = "\t".join(list(map(str, row.__dict__.values())))
                print(line, file=file, end=os.linesep)
    else:
        with open(
            file_path.parent / (file_path.stem + ".jsonl"), "w", encoding="utf-8"
        ) as file:
            for row in exp_data:
                line = json.dumps(row.__dict__, ensure_ascii=False)
                print(line, file=file, end=os.linesep)


def filter_data(
    markup_data: List[MarkupDataclass],
    aggregated_data_dict: Dict[str, str],
    dataset: str,
) -> List[AggDataclass]:
    agg_data = []
    used_wavs = set()
    for row in markup_data:
        if row.hash_id in used_wavs:
            continue
        if row.hash_id in aggregated_data_dict:
            good_agg_row = AggDataclass(
                hash_id=row.hash_id,
                audio_path=str(Path("..", "..", dataset, row.audio_path)),
                duration=row.duration,
                emotion=aggregated_data_dict[row.hash_id],
                golden_emo=row.golden_emo,
                speaker_text=row.speaker_text,
                speaker_emo=row.speaker_emo,
                source_id=row.source_id,
            )
            agg_data.append(good_agg_row)
        used_wavs.add(row.hash_id)
    return agg_data


def make_exp_data(agg_data: List[AggDataclass]) -> List[DataForExp]:
    exp_data = []
    for row in agg_data:
        if (
            not isinstance(row.golden_emo, str) or row.golden_emo == ""
        ) and row.emotion != "other":
            exp_row = DataForExp(
                id=row.hash_id,
                tensor=str(Path("..", "..", "features", row.hash_id + ".npy")),
                wav_length=row.duration,
                label=Emotion[row.emotion.upper()].value,
                emotion=row.emotion,
            )
            exp_data.append(exp_row)
    return exp_data


def aggregate_data(
    data_path: Path, out_path: Path, use_tsv: bool, dawidskene_threshold: float
) -> None:

    markup_data = ["podcast_test", "podcast_train", "crowd_train", "crowd_test"]
    data = {}
    all_data = []
    for dataset in markup_data:
        data[dataset] = read_data_markup(
            dataset_path=Path(data_path, dataset, "raw_" + dataset),
            use_tsv=use_tsv,
        )
        all_data += data[dataset]

    data_for_agg = []
    for row in all_data:
        row_for_agg = DawidSkeneEntryDataclass(
            task=row.hash_id,
            worker=row.annotator_id,
            label=row.annotator_emo,
        )
        data_for_agg.append(row_for_agg)

    aggregated_data = get_dawidskene_pred(
        data=data_for_agg,
        threshold=dawidskene_threshold,
        meta_path=data_path / "meta.tsv",
    )

    aggregated_data_dict = {row.task: row.pred for row in aggregated_data}

    exp_data = {}
    for dataset in markup_data:
        agg_data = filter_data(
            markup_data=data[dataset],
            aggregated_data_dict=aggregated_data_dict,
            dataset=dataset,
        )
        exp_data[dataset] = make_exp_data(agg_data=agg_data)
        exp_data_to_file(
            file_path=out_path / dataset.rsplit("_", maxsplit=1)[-1] / dataset,
            exp_data=exp_data[dataset],
            use_tsv=use_tsv,
        )
        agg_data_to_file(
            file_path=out_path / "aggregated_dataset" / dataset,
            agg_data=agg_data,
            use_tsv=use_tsv,
        )
    exp_data_to_file(
        file_path=out_path / "train" / "train",
        exp_data=exp_data["podcast_train"] + exp_data["crowd_train"],
        use_tsv=use_tsv,
    )
    exp_data_to_file(
        file_path=Path(out_path / "test" / "test"),
        exp_data=exp_data["podcast_test"] + exp_data["crowd_test"],
        use_tsv=use_tsv,
    )
