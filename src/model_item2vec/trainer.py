import os
import lightning as L
import pandas as pd
import torch
import mlflow
from ray import train
from ray.train import Checkpoint
from torch import nn
from evidently.metric_preset import ClassificationPreset
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report

from model import SkipGram


class LitSkipGram(L.LightningModule):
    def __init__(
        self,
        skipgram_model: SkipGram,
        learning_rate: float = 0.001,
        l2_reg: float = 1e-5,
        log_dir: str = ".",
    ):
        super().__init__()
        self.skipgram_model = skipgram_model
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.log_dir = log_dir
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        target_items = batch["target_items"]
        context_items = batch["context_items"]
        labels = batch["labels"].float().squeeze()

        predictions = self.skipgram_model.forward(target_items, context_items)
        loss_fn = nn.BCELoss()
        loss = loss_fn(predictions, labels)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if train.get_context().get_world_size() > 1:
            train.report({"train_loss": loss.item()})
        return loss

    def validation_step(self, batch, batch_idx):
        target_items = batch["target_items"]
        context_items = batch["context_items"]
        labels = batch["labels"].float().squeeze()

        predictions = self.skipgram_model.forward(target_items, context_items)
        loss_fn = nn.BCELoss()
        loss = loss_fn(predictions, labels)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return {"val_loss": loss}

    def configure_optimizers(self):
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

    def on_validation_epoch_end(self):
        sch = self.lr_schedulers()
        if sch is not None:
            self.log("learning_rate", sch.get_last_lr()[0], on_epoch=True, logger=True, sync_dist=True)

        # Aggregate validation loss for Ray Train
        val_loss = self.trainer.callback_metrics.get("val_loss", None)
        if val_loss is not None and train.get_context().get_world_size() > 1:
            checkpoint = Checkpoint.from_directory(self.trainer.checkpoint_callback.dirpath)
            train.report({"val_loss": val_loss.item()}, checkpoint=checkpoint)

    def on_fit_end(self):
        self._log_classification_metrics(self.trainer.val_dataloaders)

    def _log_classification_metrics(self, val_loader):
        target_items, context_items, labels = [], [], []
        for _, batch_input in enumerate(val_loader):
            _target_items = batch_input["target_items"].cpu().detach().numpy()
            _context_items = batch_input["context_items"].cpu().detach().numpy()
            _labels = batch_input["labels"].cpu().detach().numpy()
            target_items.extend(_target_items)
            context_items.extend(_context_items)
            labels.extend(_labels)

        val_df = pd.DataFrame({
            "target_items": target_items,
            "context_items": context_items,
            "labels": labels,
        })

        target_items = torch.tensor(val_df["target_items"].values, device=self.device)
        context_items = torch.tensor(val_df["context_items"].values, device=self.device)
        classifications = self.skipgram_model(target_items, context_items)

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

        evidently_report_fp = f"{self.log_dir}/evidently_report_classification.html"
        os.makedirs(self.log_dir, exist_ok=True)
        try:
            classification_performance_report.save_html(evidently_report_fp)
            print(f"Đã lưu báo cáo Evidently tại: {evidently_report_fp}")
        except Exception as e:
            print(f"Không thể lưu báo cáo Evidently: {str(e)}")
            return  # Bỏ qua nếu không thể lưu file

        if mlflow.active_run():
            try:
                if not os.path.exists(evidently_report_fp):
                    print(f"File {evidently_report_fp} không tồn tại, bỏ qua ghi log artifact.")
                else:
                    mlflow.log_artifact(evidently_report_fp)
                    print(f"Đã ghi log artifact: {evidently_report_fp}")
            except Exception as e:
                print(f"Không thể ghi log artifact {evidently_report_fp}: {str(e)}")
            
            try:
                for metric_result in classification_performance_report.as_dict()["metrics"]:
                    if metric_result["metric"] == "ClassificationQualityMetric":
                        roc_auc = float(metric_result["result"]["current"]["roc_auc"])
                        mlflow.log_metric("val_roc_auc", roc_auc)
                        print(f"Đã ghi log val_roc_auc: {roc_auc}")
                    elif metric_result["metric"] == "ClassificationPRTable":
                        columns = ["top_perc", "count", "prob", "tp", "fp", "precision", "recall"]
                        table = metric_result["result"]["current"][1]
                        table_df = pd.DataFrame(table, columns=columns)
                        for _, row in table_df.iterrows():
                            prob = int(row["prob"] * 100)
                            mlflow.log_metric("val_precision_at_prob", float(row["precision"]), step=prob)
                            mlflow.log_metric("val_recall_at_prob", float(row["recall"]), step=prob)
                            print(f"Đã ghi log tại prob {prob}: precision={float(row['precision'])}, recall={float(row['recall'])}")
            except Exception as e:
                print(f"Không thể ghi log metrics phân loại vào MLflow: {str(e)}")