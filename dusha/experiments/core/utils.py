import json
from pathlib import Path

import pandas as pd


def parse_name(tsv_name):
    """
    We have names like
    f"predicts_dataset_{dataset_name}_model_{model_name}.tsv" /
    f"metrics_dataset_{dataset_name}_model_{model_name}.csv"

    Returns: dataset_name, model_name
    """

    if tsv_name.startswith("predicts_dataset_"):
        # len('predicts_dataset_') = 17
        _s = tsv_name[17:]
    elif tsv_name.startswith("metrics_dataset_"):
        _s = tsv_name[16:]
    else:
        raise ValueError(f"tsv_name is {tsv_name}")

    model_prefix_start = _s.find("_model_")
    if model_prefix_start == -1:
        raise ValueError(f"tsv_name is {tsv_name}")

    dataset_name = _s[:model_prefix_start]
    model_name = _s[model_prefix_start + len("_model_") : -4]

    return dataset_name, model_name


def raw_parse_dir(exps_path, prefix="predicts"):
    """
    Pars dir with experiments and returns dicts:
        dataset: model: path
        dataset: set of models

    Args:
        exps_path: path to dir with experiments
        prefix: 'predicts' or 'metrics' - what the function should parse
    """
    exps_path = Path(exps_path)

    # get paths to data
    glob_exp = "**/"

    if prefix == "predicts":
        glob_file = "predicts_*.tsv"
    elif prefix == "metrics":
        glob_file = "metrics_*.csv"
    else:
        raise ValueError(
            f"Get prefix = {prefix}, supports only ['predicts', 'metrics']"
        )

    data_paths = list(exps_path.glob(glob_exp + glob_file))

    data_paths = [
        p
        for p in data_paths
        if str(p.name).startswith(prefix)
        and str(p.name).find("dataset_") > -1
        and str(p.name).find("model_") > -1
    ]

    # init our structure
    # dataset: model: path_to_predict
    dataset_models_paths = {}

    # get all models for all datasets

    # dataset: set of model names
    dataset_models_dict = {}
    for curr_path in data_paths:
        dataset_name, model_name = parse_name(str(curr_path.name))
        if dataset_models_dict.get(dataset_name) is None:
            dataset_models_dict[dataset_name] = {model_name}
            dataset_models_paths[dataset_name] = {model_name: curr_path}
        else:
            dataset_models_dict[dataset_name] |= {model_name}
            dataset_models_paths[dataset_name].update({model_name: curr_path})

    return dataset_models_paths, dataset_models_dict


def collect_metrics_to_one(list_of_metrics_df):
    df = list_of_metrics_df[0]
    df.columns = ["", df.columns[-1]]
    df = df.set_index("")

    for curr_metric_df in list_of_metrics_df[1:]:
        _df = curr_metric_df
        _df.columns = ["", _df.columns[-1]]
        _df = _df.set_index("")
        df = df.join(_df)

    df = df.sort_values("f1_macro", axis=1, ascending=False)

    return df


def load_jsonl_as_df(file_name):
    data = []
    with open(file_name, "r") as file1:
        for line1 in file1:
            data.append(json.loads(line1))
    file1.close()
    df = pd.DataFrame.from_records(data)
    if "label" in df.columns:
        df.label = df.label.astype(int)

    return df
