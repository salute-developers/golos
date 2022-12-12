import copy
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .metrics import calculate_metrics


class Learner:
    def __init__(
        self,
        train_dataset,
        val_dataset,
        dataloaders,
        exp_path,
        model_name,
        model,
        batch_size,
        dump_best_checkpoints,
        dump_last_checkpoints,
        best_checkpoints_warmup,
        cuda_device="cuda:0",
    ):

        self.device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)

        self.__model_name = model_name

        self.dump_last_checkpoints = dump_last_checkpoints
        self.dump_best_checkpoints = dump_best_checkpoints
        self.best_checkpoints_warmup = best_checkpoints_warmup

        self.exp_path = Path(exp_path)
        if dump_best_checkpoints:
            self.best_checkpoints_path = self.exp_path / "best_checkpoints"
            self.best_checkpoints_path.mkdir()
        if dump_last_checkpoints:
            self.last_checkpoints_path = self.exp_path / (
                self.__model_name + "_last_checkpoints"
            )
            self.last_checkpoints_path.mkdir()

        self.batch_size = batch_size

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        print(
            "train labels",
            np.unique(self.train_dataset.df.label.values, return_counts=True),
        )
        print(
            "train weights",
            np.unique(
                self.train_dataset.df.sampling_weights.values, return_counts=True
            ),
        )

        self.dataloaders = dataloaders

        self.dataset_sizes = {
            "train": len(self.train_dataset.df),
            "validate": len(self.val_dataset.df),
        }

    def train(self, num_epochs, lr, step_size, gamma, weight_decay=0, clip_grad=False):
        comment_str_list = [
            "MODEL",
            self.__model_name,
            "EPOCHS",
            str(num_epochs),
            "LR",
            str(lr),
            "BATCH",
            str(self.batch_size),
        ]

        comment_str = "_".join(comment_str_list)
        summary_writer = SummaryWriter(log_dir=self.exp_path / 'TB_log' / comment_str)

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        since = time.time()
        # copy.deepcopy(self.model.state_dict())
        best_model_wts = None
        best_loss = 10000000
        best_acc = best_f1 = best_WA = 0
        softmax = nn.Softmax(dim=1)

        try:
            for epoch in range(1, num_epochs + 1):
                print(f"Epoch {epoch}/{num_epochs}")
                for phase in ["train", "validate"]:
                    if phase == "train":
                        self.model.train()
                        cur_step_lr = scheduler.get_last_lr()[-1]
                    else:
                        self.model.eval()

                    running_loss = 0.0
                    running_outputs = []
                    running_labels = []
                    for inputs, labels in tqdm(self.dataloaders[phase]):
                        inputs = inputs.to(self.device)
                        labels = labels.long()
                        labels = labels.to(self.device)
                        optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == "train"):
                            outputs = self.model(inputs)
                            probs = softmax(outputs)
                            loss = criterion(outputs, labels)
                            if phase == "train":
                                loss.backward()
                                if clip_grad:
                                    torch.nn.utils.clip_grad_norm_(
                                        self.model.parameters(), 1.0
                                    )
                                optimizer.step()

                        running_loss += loss.item()
                        if phase == "validate":
                            running_labels.append(labels)
                            running_outputs.append(probs)

                    if phase == "train":
                        scheduler.step()

                    epoch_loss = running_loss / self.dataset_sizes[phase]
                    if phase == "validate":
                        pred_class = np.argmax(
                            torch.cat(running_outputs).cpu().numpy(), axis=1
                        )
                        gt_class = torch.cat(running_labels).cpu().numpy()

                        metric_dict = calculate_metrics(
                            pred_class, gt_class, neg_label=0
                        )

                        summary_writer.add_scalar("Loss/validate", epoch_loss, epoch)
                        for metric_name, metric_value in metric_dict.items():
                            summary_writer.add_scalar(
                                f"Metrics/{metric_name}", metric_value, epoch
                            )

                        epoch_acc = metric_dict["accuracy"]
                        epoch_f1 = metric_dict["f1_macro"]
                        epoch_WA = metric_dict["WA"]

                        print(f"{phase} Loss: {epoch_loss:.4f}")
                        print(f"{phase} Acc: {epoch_acc:.4f}")
                        print(f"{phase} F1 macro: {epoch_f1:.4f}")
                        print(f"{phase} WA: {epoch_WA:.4f}")

                        if epoch_f1 > best_f1:
                            best_f1 = epoch_f1
                            # best_WA = epoch_WA
                            best_acc = epoch_acc
                            best_f1 = epoch_f1

                            best_epoch = epoch
                            best_model_wts = copy.deepcopy(self.model.state_dict())

                            if (
                                self.dump_best_checkpoints
                                and epoch > self.best_checkpoints_warmup
                            ):
                                torch.save(
                                    best_model_wts,
                                    self.best_checkpoints_path
                                    / f"best_checkpoint_{epoch}",
                                )

                        if self.dump_last_checkpoints and abs(epoch - num_epochs) < 6:
                            torch.save(
                                copy.deepcopy(self.model.state_dict()),
                                self.last_checkpoints_path / f"checkpoint_{epoch}",
                            )

                    else:
                        print(f"{phase} Loss: {epoch_loss:.4f}")
                        summary_writer.add_scalar("Loss/train", epoch_loss, epoch)
                        summary_writer.add_scalar("LR/value", cur_step_lr, epoch)

        except KeyboardInterrupt:
            pass

        summary_writer.flush()
        time_elapsed = time.time() - since
        print(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s."
            + f" Best model loss: {best_loss:.6f}, best model acc: {best_acc:.6f}, "
            + f"best model f1: {best_f1:.6f}, best epoch {best_epoch}"
        )

        self.model.load_state_dict(best_model_wts)
        self.model.eval()
        return best_model_wts
