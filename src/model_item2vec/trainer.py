import os
from typing import Any, Dict, List, Optional

import lightning as L
import mlflow
import pandas as pd
import torch
from torch import nn
from evidently.metric_preset import ClassificationPreset
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from model_item2vec.model import SkipGram
from ray import train
from ray.train import Checkpoint


class LitSkipGram(L.LightningModule):
    """Lightning module for training a SkipGram model with distributed support and evaluation metrics."""

    def __init__(
        self,
        skipgram_model: SkipGram,
        learning_rate: float = 0.001,
        l2_reg: float = 1e-5,
        log_dir: str = ".",
    ):
        """Initialize the LitSkipGram model.

        Args:
            skipgram_model (SkipGram): The SkipGram model instance.
            learning_rate (float): Learning rate for the optimizer. Defaults to 0.001.
            l2_reg (float): L2 regularization (weight decay) for the optimizer. Defaults to 1e-5.
            log_dir (str): Directory to save evaluation reports and logs. Defaults to current directory.
        """
        super().__init__()
        self.skipgram_model = skipgram_model
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.log_dir = log_dir
        self.save_hyperparameters()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a training step on a batch of data.

        Args:
            batch (Dict[str, torch.Tensor]): Batch containing target items, context items, and labels.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        target_items = batch["target_items"]
        context_items = batch["context_items"]
        labels = batch["labels"].float().squeeze()

        predictions = self.skipgram_model.forward(target_items, context_items)
        loss_fn = nn.BCELoss()
        loss = loss_fn(predictions, labels)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        if train.get_context().get_world_size() > 1:
            train.report({"train_loss": loss.item()})
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Perform a validation step on a batch of data.

        Args:
            batch (Dict[str, torch.Tensor]): Batch containing target items, context items, and labels.
            batch_idx (int): Index of the current batch.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the validation loss.
        """
        target_items = batch["target_items"]
        context_items = batch["context_items"]
        labels = batch["labels"].float().squeeze()

        predictions = self.skipgram_model.forward(target_items, context_items)
        loss_fn = nn.BCELoss()
        loss = loss_fn(predictions, labels)

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return {"val_loss": loss}

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure the optimizer and learning rate scheduler.

        Returns:
            Dict[str, Any]: Dictionary containing the optimizer and learning rate scheduler configuration.
        """
        optimizer = torch.optim.Adam(
            self.skipgram_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_reg,
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.3, patience=2
            ),
            "monitor": "val_loss",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_validation_epoch_end(self) -> None:
        """Log the learning rate and report validation metrics for distributed training.

        Logs the current learning rate and, in distributed settings, reports the validation loss
        and checkpoint to Ray Train.
        """
        sch = self.lr_schedulers()
        if sch is not None:
            self.log(
                "learning_rate",
                sch.get_last_lr()[0],
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )

        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is not None and train.get_context().get_world_size() > 1:
            checkpoint = Checkpoint.from_directory(self.trainer.checkpoint_callback.dirpath)
            train.report({"val_loss": val_loss.item()}, checkpoint=checkpoint)

    def on_fit_end(self) -> None:
        """Log classification metrics at the end of training.

        Generates and saves an Evidently classification report and logs metrics to MLflow.
        """
        if hasattr(self.trainer, "val_dataloaders"):
            self._log_classification_metrics(self.trainer.val_dataloaders)

    def _log_classification_metrics(self, val_loader) -> None:
        """Generate and log classification metrics using Evidently and MLflow.

        Args:
            val_loader: Validation data loader containing batches of target items, context items, and labels.
        """
        target_items, context_items, labels = [], [], []
        for batch_input in val_loader:
            target_items.extend(batch_input["target_items"].cpu().detach().numpy())
            context_items.extend(batch_input["context_items"].cpu().detach().numpy())
            labels.extend(batch_input["labels"].cpu().detach().numpy())

        val_df = pd.DataFrame(
            {
                "target_items": target_items,
                "context_items": context_items,
                "labels": labels,
            }
        )

        target_items_tensor = torch.tensor(val_df["target_items"].values, device=self.device)
        context_items_tensor = torch.tensor(val_df["context_items"].values, device=self.device)
        classifications = self.skipgram_model(target_items_tensor, context_items_tensor)

        eval_classification_df = val_df.assign(
            classification_proba=classifications.cpu().detach().numpy(),
            label=lambda df: df["labels"].astype(int),
        )

        target_col = "label"
        prediction_col = "classification_proba"
        column_mapping = ColumnMapping(target=target_col, prediction=prediction_col)
        classification_performance_report = Report(metrics=[ClassificationPreset()])
        classification_performance_report.run(
            reference_data=None,
            current_data=eval_classification_df[[target_col, prediction_col]],
            column_mapping=column_mapping,
        )

        evidently_report_fp = os.path.join(self.log_dir, "evidently_report_classification.html")
        os.makedirs(self.log_dir, exist_ok=True)
        try:
            classification_performance_report.save_html(evidently_report_fp)
            print(f"Saved Evidently report to: {evidently_report_fp}")
        except Exception as e:
            print(f"Failed to save Evidently report: {str(e)}")
            return

        if mlflow.active_run():
            try:
                if not os.path.exists(evidently_report_fp):
                    print(f"File {evidently_report_fp} does not exist, skipping artifact logging.")
                else:
                    mlflow.log_artifact(evidently_report_fp)
                    print(f"Logged artifact: {evidently_report_fp}")
            except Exception as e:
                print(f"Failed to log artifact {evidently_report_fp}: {str(e)}")

            try:
                for metric_result in classification_performance_report.as_dict()["metrics"]:
                    if metric_result["metric"] == "ClassificationQualityMetric":
                        roc_auc = float(metric_result["result"]["current"]["roc_auc"])
                        mlflow.log_metric("val_roc_auc", roc_auc)
                        print(f"Logged val_roc_auc: {roc_auc}")
                    elif metric_result["metric"] == "ClassificationPRTable":
                        columns = ["top_perc", "count", "prob", "tp", "fp", "precision", "recall"]
                        table = metric_result["result"]["current"][1]
                        table_df = pd.DataFrame(table, columns=columns)
                        for _, row in table_df.iterrows():
                            prob = int(row["prob"] * 100)
                            mlflow.log_metric("val_precision_at_prob", float(row["precision"]), step=prob)
                            mlflow.log_metric("val_recall_at_prob", float(row["recall"]), step=prob)
                            print(f"Logged at prob {prob}: precision={float(row['precision'])}, recall={float(row['recall'])}")
            except Exception as e:
                print(f"Failed to log classification metrics to MLflow: {str(e)}")