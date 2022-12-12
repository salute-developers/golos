import os
from pathlib import Path

import click
import lazycon
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from core.metrics import get_metrics_df
from core.model import AUDIO_COLS, SoftMaxModel
from core.utils import collect_metrics_to_one, load_jsonl_as_df, raw_parse_dir

DEVICE = "cuda:0"


def run_single_inf(exp_path, test_manifest, with_metrics, recalculate, device):
    # parse exp_path
    # it may be exp path or path to model
    if os.path.isdir(exp_path):
        dir_path = exp_path
        _path = Path(exp_path)
        model_path = _path / _path.name
    else:
        dir_path = os.path.dirname(exp_path)
        model_path = Path(exp_path)

    dir_path = Path(dir_path)
    model_name = model_path.name

    # check the config
    config_path = dir_path / "train.config"
    assert os.path.exists(config_path), f"No train.config in {dir_path}"

    # check the model
    if not os.path.exists(model_path):
        print(f"There is no saved model {model_path}. Nothing to inference")
        return None

    # load the model
    cfg = lazycon.load(config_path)
    model = cfg.model
    try:
        model.to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    except Exception as exception:
        print(f"Problem with loading model {model_path}. Skipped")
        print(exception)
        return None

    # add last layer SoftMax to predict probabilities
    model = SoftMaxModel(model)

    # create predicts and metrics paths
    predicts_path = Path(dir_path) / "predicts"
    metrics_path = Path(dir_path) / "metrics"

    predicts_path.mkdir(exist_ok=True)
    metrics_path.mkdir(exist_ok=True)

    # parse --vm folder/df
    paths_to_inf = []

    if os.path.isdir(test_manifest):
        paths_to_inf = list(Path(test_manifest).glob("*.jsonl"))
    else:
        paths_to_inf.append(test_manifest)

    assert len(paths_to_inf) > 0, f"No .jsonl here: {test_manifest}"

    # iterate over datasets for inference
    for dataset_df_path in paths_to_inf:
        dataset_df_path = Path(dataset_df_path)
        dataset_df = load_jsonl_as_df(dataset_df_path)
        # dataset_df = pd.read_csv(dataset_df_path, sep='\t')
        dataset_name = str(dataset_df_path.name).split(".", maxsplit=1)[0]
        if with_metrics:
            assert (
                "label" in dataset_df.columns
            ), f"{dataset_name} hasn't 'label' column, but --with_metrics"

        # predict
        predicts_tsv_path = (
            predicts_path / f"predicts_dataset_{dataset_name}_model_{model_name}.tsv"
        )

        # if predicts exist and we don't want to recalculate it, but want to calculate metrics
        if os.path.exists(predicts_tsv_path) and not recalculate:
            if with_metrics:
                metrics_csv_path = (
                    metrics_path
                    / f"metrics_dataset_{dataset_name}_model_{model_name}.csv"
                )
                if not os.path.exists(metrics_csv_path):
                    print(
                        f"Predicts for {model_name} {dataset_name} exist. Calculating metrics"
                    )
                    pred_df = pd.read_csv(predicts_tsv_path, sep="\t")

                    pred_class = pred_df[AUDIO_COLS[0]].values
                    gt_class = pred_df["label"].values

                    metrics_df = get_metrics_df(
                        pred_class=pred_class, gt_class=gt_class, model_name=model_name
                    )

                    metrics_df.to_csv(metrics_csv_path)
                else:
                    print(
                        f"Predicts and metrics for {model_name} {dataset_name} exist. Skipped"
                    )
            else:
                print(
                    f"Predicts for {model_name} {dataset_name} are existed"
                    + "--no_metrics, so metrics calculation is skipped"
                )
            continue

        # calculate predicts
        running_outputs = []
        ds = cfg.get_val_dataset(_df=dataset_df, ds_base_path=dataset_df_path.parent)
        dataloader = cfg.get_val_dataloader(val_ds=ds)

        print(f"Calculating predicts and metrics: {model_name} {dataset_name}")
        for inputs, _ in tqdm(dataloader):
            inputs = inputs.to(device)
            with torch.no_grad():
                probs = model(inputs)

            running_outputs.append(probs)

        # MelEmotionsDataset changes order in df, so we should match predicts by id
        _df = ds.df.copy()
        pred_class = np.argmax(torch.cat(running_outputs).cpu().numpy(), axis=1)
        probas = torch.cat(running_outputs).cpu().numpy()

        _df[AUDIO_COLS[0]] = pred_class
        for i in range(4):
            _df[AUDIO_COLS[i + 1]] = probas[:, i]

        # match preds by id
        pred_df = dataset_df.copy()
        _df = _df.set_index("id").loc[pred_df.id]
        for _col in AUDIO_COLS:
            pred_df[_col] = _df[_col].values

        pred_df.to_csv(predicts_tsv_path, index=False, sep="\t")

        # calculate metrics
        if with_metrics:
            metrics_csv_path = (
                metrics_path / f"metrics_dataset_{dataset_name}_model_{model_name}.csv"
            )

            pred_class = pred_df[AUDIO_COLS[0]].values
            gt_class = pred_df["label"].values

            metrics_df = get_metrics_df(
                pred_class=pred_class, gt_class=gt_class, model_name=model_name
            )

            metrics_df.to_csv(metrics_csv_path)


@click.command()
@click.option(
    "-exps_path",
    "--exps_path",
    required=True,
    type=click.Path(exists=True),
    help="path folder with experiment folders (the experiment folder must have train.config file in)",
)
@click.option(
    "-vm",
    "--test_manifest",
    required=True,
    type=click.Path(exists=True),
    help="path to JSONL file/dir of JSONLs to inference",
)
@click.option(
    "--with_metrics/--no_metrics",
    default=True,
    help="calculate metrics for experiments",
)
@click.option(
    "--recalculate/--no_recalculate",
    default=False,
    help="recalculate existed predicts and metrics",
)
@click.option(
    "--recalculate_dataset_metrics/--no_dataset_metrics",
    default=True,
    help="recalculate existed grouped by dataset metrics",
)
@click.option(
    "-device", "--device", type=click.STRING, default=DEVICE, help="device to inference"
)
def run_inf(
    exps_path,
    test_manifest,
    with_metrics,
    recalculate,
    recalculate_dataset_metrics,
    device,
):
    # parse folder, find experiments folders
    exps_path = Path(exps_path)
    experiment_paths = [p.parent for p in exps_path.glob("**/train.config")]

    # predict and calc metrics for a single experiment
    for exp_path in experiment_paths:
        run_single_inf(
            exp_path=exp_path,
            test_manifest=test_manifest,
            with_metrics=with_metrics,
            recalculate=recalculate,
            device=device,
        )

    # aggregate metrics
    metrics_dump_dir = exps_path / "metrics"
    metrics_dump_dir.mkdir(exist_ok=True)

    if recalculate_dataset_metrics:
        print("Aggregating metrics")
        dataset_models_paths, dataset_models = raw_parse_dir(
            exps_path=exps_path, prefix="metrics"
        )
        datasets = sorted(dataset_models.keys())
        for dataset_name in datasets:
            metric_dump_dir = metrics_dump_dir / f"exps_{dataset_name}.csv"
            metric_df = collect_metrics_to_one(
                [
                    pd.read_csv(metrics_df_path)
                    for metrics_df_path in dataset_models_paths[dataset_name].values()
                ]
            ).T
            metric_df.to_csv(metric_dump_dir)
    else:
        print("--no_dataset_metrics, so metrics grouped by dataset are skipped")

    agg_metrics_paths = list(metrics_dump_dir.glob("*.csv"))
    if len(agg_metrics_paths) == 0:
        print("There is no grouped by dataset metrics")
    else:
        for agg_metrics_path in agg_metrics_paths:
            # remove exps_ and .csv in aggregated metrics df name
            dataset_name = str(agg_metrics_path.name)[5:-4]
            metric_df = pd.read_csv(agg_metrics_path).set_index("Unnamed: 0")
            metric_df.index.name = ""
            print("DATASET: ", dataset_name)
            print(metric_df)
            print("------------------------------------------------")


if __name__ == "__main__":
    run_inf()
