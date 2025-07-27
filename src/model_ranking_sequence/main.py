import logging
import os
import sys
from datetime import datetime

import boto3
import dill
import lightning as L
import mlflow
import numpy as np
import pandas as pd
import ray
import torch
import yaml
from dataset import UserItemBinaryDFDataset
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking import MlflowClient
from model import Ranker
from ray import tune
from ray.air import session
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from ray.tune import CLIReporter
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.stopper import TrialPlateauStopper
from torch.utils.data import DataLoader

# Import from src (assuming these are available)
from trainer import LitRanker

from data_prep_utils import chunk_transform

sys.path.insert(0, "..")
from id_mapper import IDMapper

# Load configuration
with open("../../configs/ranking_sequence.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Environment setup
AWS_KEY = cfg["aws"]["aws_access_key_id"]
AWS_SECRET = cfg["aws"]["aws_secret_access_key"]
MINIO_EP = cfg["aws"]["mlflow_s3_endpoint_url"]
MLFLOW_URI = cfg["aws"]["mlflow_tracking_uri"]
RAY_ADDRESS = cfg["ray"]["address"]

# Patch MLflow S3 client for MinIO
boto_session = boto3.session.Session(
    aws_access_key_id=AWS_KEY,
    aws_secret_access_key=AWS_SECRET,
)
S3ArtifactRepository._get_s3_client = lambda self: boto_session.client(
    "s3", endpoint_url=MINIO_EP
)

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Champion-model utilities
def find_champion_version(model_name: str) -> str:
    client = MlflowClient()
    versions = client.search_model_versions(f"name = '{model_name}'")
    for v in versions:
        if v.tags.get(cfg["item2vec"]["champion_tag_key"], "").lower() == "true":
            return v.version
    return max(versions, key=lambda v: int(v.version)).version if versions else None


def load_champion_model(model_name: str, idm_path: str):
    mlflow.set_tracking_uri(MLFLOW_URI)
    version = find_champion_version(model_name)
    if version is None:
        raise ValueError(f"No model version found for {model_name}")
    uri = f"models:/{model_name}/{version}"
    model = mlflow.pytorch.load_model(model_uri=uri)
    model.eval()
    idm = IDMapper().load(idm_path)
    if idm is None:
        raise ValueError(f"Failed to load IDMapper from {idm_path}")
    return model, idm


def update_champion_model(client, model_name, this_version, this_val_loss):
    is_champion = True
    worse = []
    for v in client.search_model_versions(f"name='{model_name}'"):
        if v.version == this_version.version:
            continue
        try:
            other_loss = client.get_run(v.run_id).data.metrics.get("final.val_loss")
            if other_loss is not None and other_loss < this_val_loss:
                is_champion = False
                break
            if other_loss is not None:
                worse.append(v)
        except Exception:
            pass

    if is_champion:
        client.set_model_version_tag(
            name=model_name,
            version=this_version.version,
            key=cfg["item2vec"]["champion_tag_key"],
            value="true",
        )
        for v in worse:
            try:
                client.delete_model_version_tag(
                    name=model_name,
                    version=v.version,
                    key=cfg["item2vec"]["champion_tag_key"],
                )
            except Exception:
                pass


def train_func(config, is_final_training=False):
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_URI
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_EP

    mlflow_run = None
    model_name = f"{config['experiment_name']}_sequence_rating"

    # Initialize MLflow
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        exp_name = config["experiment_name"]
        if mlflow.get_experiment_by_name(exp_name) is None:
            mlflow.create_experiment(exp_name)
        mlflow.set_experiment(exp_name)

        tags = {
            "phase": "final" if is_final_training else "tuning",
            **cfg["mlflow"]["experiment_tags"],
        }
        run_name = (
            f"Final Model - SequenceRating - {datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if is_final_training
            else f"Tuning Run - Trial {tune.get_trial_id()}"
        )
        mlflow_run = mlflow.start_run(run_name=run_name, tags=tags)

        mlflow.log_params(
            {
                f"{'final' if is_final_training else 'tune'}.{k}": v
                for k, v in config.items()
            }
        )
        mlflow.log_params(
            {
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": (
                    torch.version.cuda if torch.cuda.is_available() else "N/A"
                ),
            }
        )

        mlflow_logger = MLFlowLogger(
            experiment_name=exp_name,
            tracking_uri=MLFLOW_URI,
            run_id=mlflow_run.info.run_id,
        )
    except Exception as e:
        logger.error("MLflow init failed: %s", e)
        mlflow_logger = None

    try:
        # Handle top_K / top_k
        config["top_K"] = int(
            config.get("top_K", config.get("top_k", cfg["training"]["top_k"]))
        )
        config["top_k"] = int(config.get("top_k", config["top_K"]))

        # Load data
        train_df = pd.read_parquet(config["train_data_path"])
        val_df = pd.read_parquet(config["val_data_path"])
        train_df = train_df.drop(columns=["user_id", "parent_asin"], errors="ignore")
        val_df = val_df.drop(columns=["user_id", "parent_asin"], errors="ignore")

        # Load item feature pipeline
        with open(config["item_metadata_pipeline_fp"], "rb") as f:
            item_metadata_pipeline = dill.load(f)

        # Define required columns for item_metadata_pipeline
        required_columns = cfg["dataset"]["required_columns"]

        # Fill missing columns with default values
        for col in required_columns:
            if col not in train_df.columns:
                train_df[col] = 0.0
            if col not in val_df.columns:
                val_df[col] = 0.0

        # Transform item features
        train_item_features = chunk_transform(
            train_df, item_metadata_pipeline, chunk_size=10000
        )
        train_item_features = train_item_features.astype(np.float32)
        val_item_features = chunk_transform(
            val_df, item_metadata_pipeline, chunk_size=10000
        )
        val_item_features = val_item_features.astype(np.float32)

        # Validate feature sizes
        logger.info(
            "train_df rows: %d, train_item_features rows: %d",
            len(train_df),
            train_item_features.shape[0],
        )
        logger.info(
            "val_df rows: %d, val_item_features rows: %d",
            len(val_df),
            val_item_features.shape[0],
        )
        assert (
            len(train_df) == train_item_features.shape[0]
        ), f"train_item_features mismatch: {len(train_df)} vs {train_item_features.shape[0]}"
        assert (
            len(val_df) == val_item_features.shape[0]
        ), f"val_item_features mismatch: {len(val_df)} vs {val_item_features.shape[0]}"

        # Load IDMapper
        idm = IDMapper().load(config["idm_path"])
        logger.info("Number of items in idm.item_to_index: %d", len(idm.item_to_index))

        # Debug: Check consistency of item indices
        train_items = set(train_df[config["item_col"]].unique())
        val_items = set(val_df[config["item_col"]].unique())
        idm_items = set(idm.item_to_index.values())
        logger.info("Number of unique items in train_df: %d", len(train_items))
        logger.info("Number of unique items in val_df: %d", len(val_items))
        logger.info("Number of unique items in idm_items: %d", len(idm_items))
        missing_in_idm = (train_items | val_items) - idm_items
        missing_in_data = idm_items - (train_items | val_items)
        if missing_in_idm:
            logger.warning("Items in train/val but not in IDMapper: %s", missing_in_idm)
        if missing_in_data:
            logger.warning(
                "Items in IDMapper but not in train/val: %s", missing_in_data
            )

        # Validate user/item cols
        for df, name in [(train_df, "train"), (val_df, "val")]:
            if (
                config["user_col"] not in df.columns
                or config["item_col"] not in df.columns
            ):
                logger.error(
                    "%s df missing user_col=%s or item_col=%s",
                    name,
                    config["user_col"],
                    config["item_col"],
                )
                raise ValueError(
                    f"Missing {config['user_col']} or {config['item_col']} in {name} df"
                )
            df[config["user_col"]] = df[config["user_col"]].astype("int64")
            df[config["item_col"]] = df[config["item_col"]].astype("int64")
            if (
                df[config["user_col"]].isnull().any()
                or df[config["item_col"]].isnull().any()
            ):
                logger.error("%s df has null values in user_col or item_col", name)
                raise ValueError(f"Null values in user/item cols in {name} df")

        # Validate item_sequence
        for df, name in [(train_df, "train"), (val_df, "val")]:
            if "item_sequence" not in df.columns:
                logger.error("%s df missing item_sequence column", name)
                raise ValueError(f"Missing item_sequence in {name} df")
            df["item_sequence"] = df["item_sequence"].apply(
                lambda x: (
                    [int(i) for i in x] if isinstance(x, (list, np.ndarray)) else x
                )
            )
            if df["item_sequence"].isnull().any():
                logger.error("%s df has null values in item_sequence", name)
                raise ValueError(f"Null values in item_sequence in {name} df")

        # Timestamp col
        req = config.get("timestamp_col", "")
        ts_col = (
            req
            if req in train_df.columns
            else [c for c in train_df.columns if "timestamp" in c.lower()]
        )
        if not ts_col:
            logger.error("No timestamp column found in train_df")
            raise ValueError("No timestamp column found")
        ts_col = ts_col[0] if isinstance(ts_col, list) else ts_col

        # Verify required cols
        required_cols = [
            config["user_col"],
            config["item_col"],
            config["rating_col"],
            ts_col,
            "item_sequence",
        ]
        for c in required_cols:
            for df, name in [(train_df, "train"), (val_df, "val")]:
                if c not in df.columns:
                    logger.error("Missing column %s in %s df", c, name)
                    raise ValueError(f"Missing column {c} in {name} df")

        # Vocab size
        num_users = len(idm.user_to_index) + 1
        num_items = len(idm.item_to_index) + 1
        logger.info("Number of users: %d", num_users)
        logger.info("Number of items (including padding): %d", num_items)

        # Load champion model + embedding
        champ_model, idm = load_champion_model(
            config["item2vec_model_name"], config["idm_path"]
        )
        emb = champ_model.embeddings
        if emb is None:
            logger.error("champ_model.embeddings is None")
            raise ValueError("champ_model.embeddings is None")
        emb_dim = emb.embedding_dim
        logger.info("Embedding dimension: %d", emb_dim)

        # Copy embeddings from item2vec
        emb_weight = emb.weight.detach().clone()
        if emb_weight is None:
            logger.error("emb.weight is None")
            raise ValueError("emb.weight is None")
        logger.info("Original emb_weight shape: %s", emb_weight.shape)
        new_emb = torch.nn.Embedding(num_items, emb_dim, padding_idx=num_items - 1)
        if emb_weight.shape[0] < num_items:
            padding = torch.zeros(num_items - emb_weight.shape[0], emb_dim)
            emb_weight = torch.cat([emb_weight, padding], dim=0)
        elif emb_weight.shape[0] > num_items:
            emb_weight = emb_weight[:num_items]
        new_emb.weight.data.copy_(emb_weight)
        logger.info("New item embedding shape: %s", new_emb.weight.shape)

        # Create all_items_df from idm.item_to_index
        all_items = list(idm.item_to_index.values()) + [
            num_items - 1
        ]  # Include padding item
        logger.info(
            "Number of items in all_items (including padding): %d", len(all_items)
        )
        all_items_df = pd.DataFrame({config["item_col"]: all_items})

        # Combine train_df and val_df to get item metadata
        item_metadata_df = pd.concat([train_df, val_df]).drop_duplicates(
            subset=[config["item_col"]]
        )
        logger.info(
            "Number of unique items in item_metadata_df: %d", len(item_metadata_df)
        )

        # Debug: Check for items in idm but not in item_metadata_df
        missing_in_metadata = set(all_items) - set(item_metadata_df[config["item_col"]])
        if missing_in_metadata:
            logger.warning(
                "Items in all_items but not in item_metadata_df: %s",
                missing_in_metadata,
            )

        # Perform merge
        logger.info("Size of all_items_df before merge: %d", len(all_items_df))
        all_items_df = all_items_df.merge(
            item_metadata_df, on=config["item_col"], how="left"
        )
        logger.info("Size of all_items_df after merge: %d", len(all_items_df))

        # Debug: Check for missing items after merge
        actual_items = set(all_items_df[config["item_col"]])
        missing_items = set(all_items) - actual_items
        extra_items = actual_items - set(all_items)
        if missing_items:
            logger.error(
                "Items missing from all_items_df after merge: %s", missing_items
            )
            raise ValueError(f"Missing items after merge: {missing_items}")
        if extra_items:
            logger.warning("Extra items in all_items_df after merge: %s", extra_items)
            all_items_df = all_items_df[
                all_items_df[config["item_col"]].isin(all_items)
            ]
            logger.info(
                "Size of all_items_df after removing extra items: %d", len(all_items_df)
            )

        # Verify size
        if len(all_items_df) != len(all_items):
            logger.error(
                "Size mismatch: all_items_df has %d rows, expected %d",
                len(all_items_df),
                len(all_items),
            )
            raise ValueError(
                f"Size mismatch: all_items_df has {len(all_items_df)} rows, expected {len(all_items)}"
            )

        # Debug: Check for NaN values in all_items_df
        nan_columns = all_items_df.columns[all_items_df.isna().any()].tolist()
        if nan_columns:
            logger.warning("Columns with NaN in all_items_df: %s", nan_columns)
            for col in nan_columns:
                nan_count = all_items_df[col].isna().sum()
                nan_rows = all_items_df[all_items_df[col].isna()][
                    config["item_col"]
                ].tolist()
                logger.info(
                    "Column %s has %d NaN values in rows with item IDs: %s",
                    col,
                    nan_count,
                    nan_rows,
                )

        # Fill NaN for all columns
        for col in all_items_df.columns:
            if all_items_df[col].dtype == object:
                all_items_df[col] = all_items_df[col].fillna("")
            else:
                all_items_df[col] = all_items_df[col].fillna(0.0)

        logger.info("Filled NaN with 0.0 for all columns in all_items_df")

        # Verify no NaN remains
        if all_items_df.isna().any().any():
            logger.error("NaN values still present in all_items_df after fillna")
            raise ValueError("NaN values still present in all_items_df after fillna")

        # Transform all_items_features
        all_items_indices = all_items_df[config["item_col"]].values
        logger.info(
            "Length of all_items_indices before transform: %d", len(all_items_indices)
        )
        try:
            all_items_features = chunk_transform(
                all_items_df, item_metadata_pipeline, chunk_size=10000
            )
            all_items_features = all_items_features.astype(np.float32)
        except Exception as e:
            logger.error("Failed to transform all_items_features: %s", e)
            raise
        logger.info("Shape of all_items_features: %s", all_items_features.shape)
        logger.info(
            "Length of all_items_indices after transform: %d", len(all_items_indices)
        )

        # Debug: Validate sizes
        if len(all_items_indices) != num_items:
            logger.error(
                "Mismatch in all_items_indices: %d vs %d",
                len(all_items_indices),
                num_items,
            )
            raise ValueError(
                f"Mismatch in all_items_indices: {len(all_items_indices)} vs {num_items}"
            )
        if all_items_features.shape[0] != num_items:
            logger.error(
                "Mismatch in all_items_features: %d vs %d",
                all_items_features.shape[0],
                num_items,
            )
            raise ValueError(
                f"Mismatch in all_items_features: {all_items_features.shape[0]} vs {num_items}"
            )

        # DataLoaders
        def collate_fn(batch):
            if not batch:
                logger.error("Empty batch")
                raise ValueError("Empty batch")
            batch_dict = {
                "user": torch.stack([x["user"] for x in batch]),
                "item_sequence": torch.stack([x["item_sequence"] for x in batch]),
                "item": torch.stack([x["item"] for x in batch]),
                "rating": torch.stack([x["rating"] for x in batch]),
                "item_sequence_ts_bucket": torch.stack(
                    [x["item_sequence_ts_bucket"] for x in batch]
                ),
                "item_feature": torch.stack([x["item_feature"] for x in batch]),
            }
            batch_size = batch_dict["user"].shape[0]
            for key, tensor in batch_dict.items():
                assert (
                    tensor.shape[0] == batch_size
                ), f"Mismatch in {key} batch size: {tensor.shape[0]} vs {batch_size}"
            return batch_dict

        train_loader = DataLoader(
            UserItemBinaryDFDataset(
                train_df,
                config["user_col"],
                config["item_col"],
                config["rating_col"],
                ts_col,
                item_feature=train_item_features,
            ),
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=int(config["num_workers"]),
            collate_fn=collate_fn,
            pin_memory=True,
        )
        val_loader = DataLoader(
            UserItemBinaryDFDataset(
                val_df,
                config["user_col"],
                config["item_col"],
                config["rating_col"],
                ts_col,
                item_feature=val_item_features,
            ),
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config["num_workers"]),
            collate_fn=collate_fn,
            pin_memory=True,
        )

        # Initialize model
        model = Ranker(
            num_users,
            num_items,
            emb_dim,
            item_sequence_ts_bucket_size=cfg["training"][
                "item_sequence_ts_bucket_size"
            ],
            bucket_embedding_dim=cfg["training"]["bucket_embedding_dim"],
            item_feature_size=train_item_features.shape[1],
            item_embedding=new_emb,
        )
        if model is None:
            logger.error("Ranker model is None")
            raise ValueError("Ranker model is None")

        args_ns = type("ArgsNS", (), {})()
        for k, v in config.items():
            setattr(args_ns, k, v)

        lit = LitRanker(
            model=model,
            learning_rate=float(config["learning_rate"]),
            l2_reg=float(config["l2_reg"]),
            log_dir=config["log_dir"],
            evaluate_ranking=is_final_training,
            idm=idm,
            all_items_indices=all_items_indices,
            all_items_features=all_items_features,
            args=args_ns,
            checkpoint_callback=None,
        )
        if lit is None:
            logger.error("LitRanker is None")
            raise ValueError("LitRanker is None")

        os.makedirs(lit.log_dir, exist_ok=True)

        # Train
        trainer = L.Trainer(
            max_epochs=int(config["max_epochs"]),
            accelerator="auto",
            callbacks=[
                L.pytorch.callbacks.ModelCheckpoint(
                    dirpath=lit.log_dir,
                    filename=cfg["tune"]["model_checkpoint"]["filename"],
                    save_top_k=cfg["tune"]["model_checkpoint"]["save_top_k"],
                    monitor=cfg["tune"]["model_checkpoint"]["monitor"],
                    mode=cfg["tune"]["model_checkpoint"]["mode"],
                ),
                EarlyStopping(
                    monitor=cfg["tune"]["early_stopping"]["monitor"],
                    patience=cfg["tune"]["early_stopping"]["patience"],
                    mode=cfg["tune"]["early_stopping"]["mode"],
                ),
            ],
            logger=mlflow_logger or False,
        )

        trainer.fit(lit, train_loader, val_loader)

        # Debug: Log available callback metrics
        logger.info("Available callback metrics: %s", trainer.callback_metrics.keys())

        # Get metrics
        val_loss = trainer.callback_metrics.get(
            "val_loss", torch.tensor(float("inf"))
        ).item()
        val_roc_auc = trainer.callback_metrics.get(
            "val_roc_auc", torch.tensor(0.0)
        ).item()

        # Report to Tune
        if not is_final_training:
            logger.info(
                "Reporting metrics to Tune: val_loss=%f, val_roc_auc=%f",
                val_loss,
                val_roc_auc,
            )
            session.report({"val_loss": val_loss, "val_roc_auc": val_roc_auc})

        # Log metrics to MLflow
        if mlflow_run:
            metrics = {
                k: float(v.item() if hasattr(v, "item") else v)
                for k, v in trainer.callback_metrics.items()
            }
            logger.info("Logging metrics to MLflow: %s", metrics)
            mlflow.log_metrics(metrics)

        # Final model logging
        if is_final_training:
            from mlflow.models.signature import ModelSignature
            from mlflow.types.schema import Schema, TensorSpec

            in_schema = Schema(
                [
                    TensorSpec(name="user", type=np.dtype("int64"), shape=(-1,)),
                    TensorSpec(name="item", type=np.dtype("int64"), shape=(-1,)),
                    TensorSpec(
                        name="item_sequence", type=np.dtype("int64"), shape=(-1, -1)
                    ),
                ]
            )
            out_schema = Schema(
                [TensorSpec(name="output", type=np.dtype("float32"), shape=(-1,))]
            )
            signature = ModelSignature(inputs=in_schema, outputs=out_schema)

            mlflow.pytorch.log_model(
                pytorch_model=lit.model,
                artifact_path="sequence_rating_model",
                registered_model_name=model_name,
                signature=signature,
                metadata=cfg["mlflow"]["model_metadata"],
            )

            client = MlflowClient()
            latest = client.get_latest_versions(model_name, stages=["None"])
            if latest:
                update_champion_model(client, model_name, latest[0], val_loss)

            id_mapper_path = os.path.join(lit.log_dir, "id_mapper.pkl")
            with open(id_mapper_path, "wb") as f:
                dill.dump(idm, f)
            mlflow.log_artifact(id_mapper_path, artifact_path="id_mapper")

            item_pipeline_path = os.path.join(lit.log_dir, "item_metadata_pipeline.pkl")
            with open(item_pipeline_path, "wb") as f:
                dill.dump(item_metadata_pipeline, f)
            mlflow.log_artifact(
                item_pipeline_path, artifact_path="item_metadata_pipeline"
            )

    except Exception as e:
        logger.error("Training failed: %s", e, exc_info=True)
        raise
    finally:
        if mlflow_run:
            mlflow.end_run()


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(cfg["experiment"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["experiment"]["storage_path"], exist_ok=True)

    current_dir = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

    ray.init(
        address=RAY_ADDRESS,
        runtime_env={
            "working_dir": current_dir,
            "py_modules": [os.path.join(parent_dir, "id_mapper.py")],
            "env_vars": {
                "PYTHONPATH": base_dir,
                "AWS_ACCESS_KEY_ID": AWS_KEY,
                "AWS_SECRET_ACCESS_KEY": AWS_SECRET,
                "MLflow_S3_IGNORE_TLS": "true",
                "MLFLOW_S3_ENDPOINT_URL": MINIO_EP,
                "MLFLOW_TRACKING_URI": MLFLOW_URI,
            },
        },
    )

    base_cfg = {
        **cfg["data"],
        **cfg["dataset"],
        **cfg["training"],
        "item2vec_model_name": cfg["item2vec"]["item2vec_model_name"],
        "idm_path": cfg["data"]["idm_path"],
        "experiment_name": cfg["experiment"]["name"],
        "log_dir": cfg["experiment"]["log_dir"],
    }

    search_space = {
        "learning_rate": tune.loguniform(
            float(cfg["tune"]["learning_rate"]["min"]),
            float(cfg["tune"]["learning_rate"]["max"]),
        ),
        "l2_reg": tune.loguniform(
            float(cfg["tune"]["l2_reg"]["min"]), float(cfg["tune"]["l2_reg"]["max"])
        ),
        "batch_size": tune.choice(cfg["tune"]["batch_size"]["values"]),
        "dropout": tune.uniform(
            float(cfg["tune"]["dropout"]["min"]), float(cfg["tune"]["dropout"]["max"])
        ),
        "scheduler_factor": tune.uniform(
            float(cfg["tune"]["scheduler_factor"]["min"]),
            float(cfg["tune"]["scheduler_factor"]["max"]),
        ),
        "scheduler_patience": tune.choice(cfg["tune"]["scheduler_patience"]["values"]),
    }

    reporter = CLIReporter(
        parameter_columns=[
            "learning_rate",
            "l2_reg",
            "batch_size",
            "dropout",
            "scheduler_factor",
            "scheduler_patience",
        ],
        metric_columns=["val_roc_auc", "val_loss"],
    )

    stopper = TrialPlateauStopper(
        metric=cfg["tune"]["metric"],
        std=cfg["tune"]["stopper"]["std"],
        num_results=cfg["tune"]["stopper"]["num_results"],
        grace_period=cfg["tune"]["stopper"]["grace_period"],
        mode=cfg["tune"]["mode"],
    )

    hyperopt_search = HyperOptSearch(
        metric=cfg["tune"]["metric"],
        mode=cfg["tune"]["mode"],
        n_initial_points=cfg["tune"]["hyperopt"]["n_initial_points"],
    )

    analysis = tune.run(
        train_func,
        config={**base_cfg, **search_space},
        num_samples=int(cfg["tune"]["num_samples"]),
        search_alg=hyperopt_search,
        metric=cfg["tune"]["metric"],
        mode=cfg["tune"]["mode"],
        progress_reporter=reporter,
        storage_path=cfg["experiment"]["storage_path"],
        name=cfg["experiment"]["name"],
        stop=stopper,
        resources_per_trial={
            "cpu": cfg["tune"]["resources_per_trial"]["cpu"],
            "gpu": (
                cfg["tune"]["resources_per_trial"]["gpu"]
                if torch.cuda.is_available()
                else 0
            ),
        },
    )
    print(">>> Best hyperparameters:", analysis.best_config)

    print("\nStarting final training...")
    final_ckpt = os.path.join(cfg["experiment"]["checkpoint_dir"], "final_model")
    os.makedirs(final_ckpt, exist_ok=True)
    final_cfg = {**base_cfg, **analysis.best_config, "log_dir": final_ckpt}

    final_trainer = TorchTrainer(
        train_loop_per_worker=lambda c: train_func(c, is_final_training=True),
        train_loop_config=final_cfg,
        scaling_config=ScalingConfig(num_workers=1, use_gpu=torch.cuda.is_available()),
    )
    final_trainer.fit()
    print("Final training completed! Checkpoint at:", final_ckpt)

    ray.shutdown()
