from pathlib import Path

import click
import numpy as np
from utils.aggregation import aggregate_data, read_data_markup
from utils.calculate_features import load_features


@click.command()
@click.option(
    "-dataset_path",
    "--dataset_path",
    required=True,
    type=click.Path(exists=True),
    help="dataset_path",
)
@click.option(
    "--use_tsv", "-tsv", is_flag=True, default=False, help="use tsv to read/write"
)
@click.option(
    "--recalculate_features",
    "-rf",
    is_flag=True,
    default=False,
    help="recalculate features",
)
@click.option(
    "--threshold",
    "-threshold",
    default=0.9,
    help="Dawidskene threshold",
    show_default=True,
)
def processing(
    dataset_path: str, use_tsv: bool, recalculate_features: bool, threshold: float
) -> None:
    """
    processing raw data for training
    """
    if threshold > 1 or threshold < 0:
        raise AttributeError

    np.seterr(divide="ignore")

    public_data = Path(dataset_path)
    result_dir = public_data / f"processed_dataset_0{int(threshold*100)}"

    path_names = ["train", "aggregated_dataset", "test"]
    for path_name in path_names:
        (result_dir / path_name).mkdir(parents=True, exist_ok=True)

    (public_data / "features").mkdir(parents=True, exist_ok=True)

    data_types = ["crowd_train", "crowd_test", "podcast_train", "podcast_test"]
    for data_type in data_types:
        wavs_path = public_data / data_type / "wavs"
        data = read_data_markup(
            dataset_path=public_data / data_type / ("raw_" + data_type),
            use_tsv=use_tsv,
        )
        wavs_names = {Path(row.audio_path).stem for row in data}
        load_features(
            wavs_path=wavs_path,
            wavs_names=wavs_names,
            result_dir=public_data,
            dataset_name=data_type,
            recalculate_feature=recalculate_features,
        )

    aggregate_data(public_data, result_dir, use_tsv, threshold)


if __name__ == "__main__":
    processing()  # pylint: disable=no-value-for-parameter
