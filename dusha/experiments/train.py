from pathlib import Path
import random
import shutil

import click
import lazycon
import numpy as np
import torch

from core.learner import Learner


@click.command()
@click.option(
    "-config",
    "--config_path",
    required=True,
    type=click.Path(exists=True),
    help="path to .config file",
)
@click.option(
    "-exp_path",
    "--exp_path",
    required=True,
    type=click.Path(),
    help="path to dump experiment",
)
def train_model(config_path, exp_path):
    exp_path = Path(exp_path)
    model_name = exp_path.name
    cfg = lazycon.load(config_path)
    base_path = cfg.base_path
    assert (
        base_path.exists()
    ), f"{base_path} doesn't exist. Correct base_path in configs/data.config"

    exp_path.mkdir(parents=True, exist_ok=True)

    # dump params
    # save compiled config
    cfg.dump(exp_path / "train.config")

    # dump jsonls
    shutil.copy(cfg.train_manifest_path, exp_path / "train.jsonl")
    shutil.copy(cfg.val_manifest_path, exp_path / "val.jsonl")

    model = cfg.model

    # load pretrained model
    if cfg.pt_model_path is not None:
        model.load_state_dict(torch.load(cfg.pt_model_path, map_location="cuda:0"))
        shutil.copy(cfg.pt_model_path, exp_path / "pt_model")

    # init learner
    learner = Learner(
        train_dataset=cfg.train_dataset,
        val_dataset=cfg.val_dataset,
        dataloaders=cfg.dataloaders,
        exp_path=exp_path,
        model_name=model_name,
        model=model,
        batch_size=cfg.batch_size,
        dump_best_checkpoints=cfg.DUMP_BEST_CHECKPOINTS,
        dump_last_checkpoints=cfg.DUMP_LAST_CHECKPOINTS,
        best_checkpoints_warmup=cfg.BEST_CHECKPOINTS_WARMUP,
    )

    # train
    best_model_wts = learner.train(
        num_epochs=cfg.epoch_count,
        lr=cfg.learning_rate,
        step_size=cfg.optimizer_step,
        gamma=cfg.optimizer_gamma,
        weight_decay=cfg.weight_decay,
        clip_grad=cfg.clip_grad,
    )

    # dump best model
    torch.save(best_model_wts, exp_path / model_name)


if __name__ == "__main__":
    # fix seeds for reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    train_model()
