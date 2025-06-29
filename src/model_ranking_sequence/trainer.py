import os
import warnings

import lightning as L
import pandas as pd
import torch
from evidently.metric_preset import ClassificationPreset
from evidently.metrics import (
    FBetaTopKMetric,
    NDCGKMetric,
    PersonalizationMetric
)
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from loguru import logger
from pydantic import BaseModel
from torch import nn
from torchmetrics import AUROC

from eval.utils import create_label_df, create_rec_df, merge_recs_with_target

from viz import color_scheme
import numpy as np
from model import Ranker
import sys
sys.path.insert(0, "..")
from id_mapper import IDMapper
warnings.filterwarnings(
    action="ignore",
    category=FutureWarning,
    module=r"evidently.metrics.recsys.precision_recall_k",
)


class LitRanker(L.LightningModule):
    def __init__(
        self,
        model: Ranker,
        learning_rate: float = 0.001,
        l2_reg: float = 1e-5,
        log_dir: str = ".",
        evaluate_ranking: bool = False,
        idm: IDMapper = None,
        all_items_indices=None,
        all_items_features=None,
        args: BaseModel = None,
        neg_to_pos_ratio: int = 3,
        checkpoint_callback=None,
        accelerator: str = "cpu",
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.log_dir = log_dir
        # Currently _log_ranking_metrics method has a lot of dependencies
        # It requires IDMapper and a bunch of other paramameters
        # TODO: Refactor
        self.evaluate_ranking = evaluate_ranking
        self.idm = idm
        self.all_items_indices = all_items_indices
        self.all_items_features = all_items_features
        self.args = args
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.checkpoint_callback = checkpoint_callback
        self.accelerator = accelerator

        # Initialize AUROC for binary classification
        self.val_roc_auc_metric = AUROC(task="binary")

    def log_weight_norms(self, stage="train"):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.log(
                    f"{stage}_weight_norm_{name}",
                    param.norm().item(),
                    on_step=False,
                    logger=True,
                )

    def training_step(self, batch, batch_idx):
        input_user_ids = batch["user"]
        input_item_ids = batch["item"]
        input_item_sequences = batch["item_sequence"]
        input_item_sequence_ts_buckets = batch["item_sequence_ts_bucket"]
        input_item_features = batch["item_feature"]

        labels = batch["rating"].float()
        predictions = self.model.forward(
            input_user_ids,
            input_item_sequences,
            input_item_sequence_ts_buckets,
            input_item_features,
            input_item_ids,
        ).view(labels.shape)
        weights = torch.where(labels == 1, self.neg_to_pos_ratio, 1.0)

        loss_fn = self._get_loss_fn(weights)
        loss = loss_fn(predictions, labels)

        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        input_user_ids = batch["user"]
        input_item_ids = batch["item"]
        input_item_sequences = batch["item_sequence"]
        input_item_sequence_ts_buckets = batch["item_sequence_ts_bucket"]
        input_item_features = batch["item_feature"]

        labels = batch["rating"]
        predictions = self.model.forward(
            input_user_ids,
            input_item_sequences,
            input_item_sequence_ts_buckets,
            input_item_features,
            input_item_ids,
        ).view(labels.shape)
        weights = torch.where(labels == 1, self.neg_to_pos_ratio, 1.0)

        loss_fn = self._get_loss_fn(weights)
        loss = loss_fn(predictions, labels)

        # Update AUROC with current batch predictions and labels
        self.val_roc_auc_metric.update(predictions, labels.int())

        # Compute current running AUROC and log it at each validation step
        current_roc_auc = self.val_roc_auc_metric.compute()
        self.log(
            "val_roc_auc",
            current_roc_auc,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss

    def on_train_epoch_end(self):
        self.log_weight_norms(stage="train")

    def on_validation_epoch_end(self):
        self.log_weight_norms(stage="val")

    def configure_optimizers(self):
        # Create the optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_reg,
        )

        # Create the scheduler
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
            self.log("learning_rate", sch.get_last_lr()[0], sync_dist=True)

        # Compute and log the final ROC-AUC for the epoch
        roc_auc = self.val_roc_auc_metric.compute()
        self.log("val_roc_auc", roc_auc, sync_dist=True)
        # Reset the metric for the next epoch
        self.val_roc_auc_metric.reset()

    def on_fit_end(self):
        if self.checkpoint_callback:
            logger.info(
                f"Loading best model from {self.checkpoint_callback.best_model_path}..."
            )
            self.model = LitRanker.load_from_checkpoint(
                self.checkpoint_callback.best_model_path, model=self.model
            ).model
        self.model = self.model.to(self._get_device())
        logger.info("Logging classification metrics...")
        self._log_classification_metrics()
        if self.evaluate_ranking:
            logger.info("Logging ranking metrics...")
            self._log_ranking_metrics()

    def _log_classification_metrics(
        self,
    ):
        # Need to call model.eval() here to disable dropout and batchnorm at prediction
        # Else the model output score would be severely affected
        self.model.eval()

        val_loader = self.trainer.val_dataloaders

        labels = []
        classifications = []

        for _, batch_input in enumerate(val_loader):
            _input_user_ids = batch_input["user"].to(self._get_device())
            _input_item_ids = batch_input["item"].to(self._get_device())
            _input_item_sequences = batch_input["item_sequence"].to(self._get_device())
            _input_item_sequence_ts_buckets = batch_input["item_sequence_ts_bucket"].to(
                self._get_device()
            )
            _input_item_features = batch_input["item_feature"].to(self._get_device())
            _labels = batch_input["rating"].to(self._get_device())
            _classifications = self.model.predict(
                _input_user_ids,
                _input_item_sequences,
                _input_item_sequence_ts_buckets,
                _input_item_features,
                _input_item_ids,
            ).view(_labels.shape)

            labels.extend(_labels.cpu().detach().numpy())
            classifications.extend(_classifications.cpu().detach().numpy())
        eval_classification_df = pd.DataFrame(
            {
                "labels": labels,
                "classification_proba": classifications,
            }
        ).assign(label=lambda df: df["labels"].gt(0).astype(int))

        self.eval_classification_df = eval_classification_df

        # Evidently
        target_col = "label"
        prediction_col = "classification_proba"
        column_mapping = ColumnMapping(target=target_col, prediction=prediction_col)
        classification_performance_report = Report(
            metrics=[
                ClassificationPreset(),
            ]
        )

        classification_performance_report.run(
            reference_data=None,
            current_data=eval_classification_df[[target_col, prediction_col]],
            column_mapping=column_mapping,
        )

        evidently_report_fp = f"{self.log_dir}/evidently_report_classification.html"
        os.makedirs(self.log_dir, exist_ok=True)
        classification_performance_report.save_html(evidently_report_fp)

        if "mlflow" in str(self.logger.__class__).lower():
            run_id = self.logger.run_id
            mlf_client = self.logger.experiment
            mlf_client.log_artifact(run_id, evidently_report_fp)
            for metric_result in classification_performance_report.as_dict()["metrics"]:
                metric = metric_result["metric"]
                if metric == "ClassificationQualityMetric":
                    roc_auc = float(metric_result["result"]["current"]["roc_auc"])
                    mlf_client.log_metric(run_id, "val_roc_auc", roc_auc)
                    continue
                if metric == "ClassificationPRTable":
                    columns = [
                        "top_perc",
                        "count",
                        "prob",
                        "tp",
                        "fp",
                        "precision",
                        "recall",
                    ]
                    table = metric_result["result"]["current"][1]
                    table_df = pd.DataFrame(table, columns=columns)
                    for i, row in table_df.iterrows():
                        prob = int(row["prob"] * 100)
                        precision = float(row["precision"])
                        recall = float(row["recall"])
                        mlf_client.log_metric(
                            run_id,
                            "val_precision_at_prob",
                            precision,
                            step=prob,
                        )
                        mlf_client.log_metric(
                            run_id,
                            "val_recall_at_prob",
                            recall,
                            step=prob,
                        )
                    break

    def _log_ranking_metrics(self):
        # ========== THAM SỐ ==========
        ts = self.args.timestamp_col
        rc = self.args.rating_col
        uc = self.args.user_col
        ic = self.args.item_col
        K = self.args.top_K
        k = self.args.top_k
        idm = self.idm

        val_loaders = self.trainer.val_dataloaders
        ds = val_loaders[0].dataset if isinstance(val_loaders, list) else val_loaders.dataset
        df = ds.df.copy()

        # ========== MAP ID TO INDEX (siêu nhanh, không dùng apply) ==========
        if df[uc].dtype != "int64":
            if hasattr(idm, "get_user_index"):
                user_map = {u: idm.get_user_index(u) for u in df[uc].unique()}
                df[uc] = df[uc].map(user_map).astype("int64")
        if df[ic].dtype != "int64":
            if hasattr(idm, "get_item_index"):
                item_map = {i: idm.get_item_index(i) for i in df[ic].unique()}
                df[ic] = df[ic].map(item_map).astype("int64")

        # ========== LẤY USER DỰ ĐOÁN ==========
        to_rec = df.sort_values(ts).drop_duplicates(subset=[uc])
        all_users = to_rec[uc].values

        # SAMPLE USER nếu quá nhiều
        MAX_USERS = 3000
        if len(all_users) > MAX_USERS:
            sel_idx = np.random.choice(len(all_users), MAX_USERS, replace=False)
            to_rec = to_rec.iloc[sel_idx]
            logger.warning(f"Too many users ({len(all_users)}), sample {MAX_USERS} users for fast evaluation!")

        user_ids = to_rec[uc].values
        item_sequences = np.stack(to_rec["item_sequence"].values)
        # ========== BUCKET ==========
        item_ts_bucket_col = [c for c in to_rec.columns if "ts_bucket" in c.lower()]
        if not item_ts_bucket_col:
            raise ValueError("No item sequence timestamp bucket column found in dataset")
        item_ts_bucket_col = item_ts_bucket_col[0]
        item_ts_buckets = np.stack(to_rec[item_ts_bucket_col].values)

        device = self._get_device()
        # ========== TENSORIZE ITEM FEATURES ==========
        item_features = torch.tensor(self.all_items_features, device=device)
        item_indices = torch.tensor(self.all_items_indices, device=device)

        # ========== BATCH RECOMMEND ==========
        self.model.eval()
        batch_size = 1024
        recs = []
        n_users = len(user_ids)
        for i in range(0, n_users, batch_size):
            u_batch = torch.tensor(user_ids[i:i+batch_size], device=device)
            seq_batch = torch.tensor(item_sequences[i:i+batch_size], device=device)
            bucket_batch = torch.tensor(item_ts_buckets[i:i+batch_size], device=device)
            with torch.no_grad():
                rec_batch = self.model.recommend(
                    u_batch,
                    seq_batch,
                    bucket_batch,
                    item_features,
                    item_indices,
                    k=K,
                    batch_size=K
                )
            if isinstance(rec_batch, torch.Tensor):
                rec_batch = rec_batch.cpu().numpy()
            recs.append(rec_batch)
        recs = np.vstack(recs)  # [num_users, K]

        # Personalization (1 - jaccard similarity giữa các user)
        def personalization_at_k(arr):
            if len(arr) < 2: return 1.0
            total = 0
            count = 0
            for i in range(len(arr)):
                for j in range(i+1, len(arr)):
                    a, b = set(arr[i]), set(arr[j])
                    if not a and not b: continue
                    total += len(a & b) / len(a | b)
                    count += 1
            return 1 - (total / count) if count > 0 else 1.0
        personalization = personalization_at_k(recs)

        # ========== LƯU DATAFRAME CHO EVIDENTLY ==========
        rec_df = pd.DataFrame({
            uc: user_ids.repeat(K),
            ic: recs.flatten(),
            "rec_ranking": np.tile(np.arange(1, K+1), len(user_ids))
        })
        # label_df cần cho evidently (item đã thực sự được user interact)
        label_df = df[[uc, ic, rc, ts]].copy()
        label_df = label_df.groupby([uc, ic], as_index=False).first()

        eval_df = pd.merge(
            rec_df, label_df, on=[uc, ic], how="left"
        )
        eval_df[rc] = eval_df[rc].fillna(0.0)
        self.eval_ranking_df = eval_df

        # ========== TẠO EVIDENTLY REPORT (KHÔNG CÓ precision@K, recall@K) ==========
        col_map = ColumnMapping(
            recommendations_type="rank",
            target=rc, prediction="rec_ranking",
            item_id=ic, user_id=uc
        )
        report = Report(
            metrics=[
                NDCGKMetric(k=k),
                FBetaTopKMetric(k=k),
                PersonalizationMetric(k=k),
            ],
            options=[color_scheme],
        )
        report.run(reference_data=None, current_data=eval_df, column_mapping=col_map)

        html_fp = f"{self.log_dir}/evidently_report_ranking.html"
        os.makedirs(self.log_dir, exist_ok=True)
        report.save_html(html_fp)

        # ========== LOG METRICS LÊN MLflow ==========
        if "mlflow" in str(self.logger.__class__).lower():
            rid = self.logger.run_id
            cli = self.logger.experiment

            cli.log_metric(rid, "val_personalization_at_K", personalization)
            cli.log_artifact(rid, html_fp)

            for m in report.as_dict()["metrics"]:
                mt = m["metric"]
                mt_clean = mt.replace("@", "_at_")
                if mt == "PersonalizationMetric":
                    cli.log_metric(rid, f"val_{mt_clean}", float(m["result"]["current_value"]))
                else:
                    for step, val in m["result"]["current"].items():
                        cli.log_metric(rid, f"val_{mt_clean}_as_step", float(val), step=int(step))

        # Nếu muốn log thêm các threshold-based metrics (precision/recall theo threshold):
        # Đoạn này tuỳ vào cách bạn implement và dùng threshold cho prediction
        # Có thể đưa từ _log_classification_metrics hoặc custom thêm.

    def _get_loss_fn(self, weights):
        return nn.BCELoss(weights)

    def _get_device(self):
        return self.accelerator
