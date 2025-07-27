import os
import sys
import time
from pathlib import Path

import lightning as L
import psutil
import ray
import torch
import yaml
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from loguru import logger
from ray import train, tune
from ray.train import CheckpointConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from ray.tune import RunConfig, Tuner
from torch.utils.data import DataLoader

# Configure loguru
logger.remove()
logger.add(sys.stderr, level="INFO")

from dataset import SkipGramDataset
from model import SkipGram
from trainer import LitSkipGram

sys.path.insert(0, "..")
from id_mapper import IDMapper


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # Convert loguniform ranges to float
    for param in ["learning_rate", "l2_reg"]:
        if "loguniform" in config["training"][param]:
            config["training"][param]["loguniform"] = [
                float(x) for x in config["training"][param]["loguniform"]
            ]
    # Resolve paths
    config["data_path"] = str(Path(config["data_path"]).resolve())
    config["checkpoint_dir"] = str(Path(config["checkpoint_dir"]).resolve())
    config["final_checkpoint_dir"] = str(Path(config["final_checkpoint_dir"]).resolve())
    config["storage_path"] = str(Path(config["storage_path"]).resolve())
    return config


def train_func(config, is_final_training=False):
    import mlflow

    print("Current working directory:", os.getcwd())
    print("Train func config:", config)

    # Initialize MLflow
    mlflow_run = None
    try:
        mlflow.set_tracking_uri(config["mlflow_tracking_uri"])

        # Create experiment hierarchy
        base_experiment = config["run_name"]
        experiment_name = (
            f"{base_experiment}/final_model"
            if is_final_training
            else f"{base_experiment}/hyperparameter_tuning"
        )

        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
        except Exception as e:
            print(f"Warning: Failed to create experiment {experiment_name}: {str(e)}")
            try:
                experiment_id = mlflow.create_experiment(experiment_name)
                print(f"Created new experiment with ID: {experiment_id}")
            except Exception as e:
                print(f"Failed to create experiment with new ID: {str(e)}")

        mlflow.set_experiment(experiment_name)

        tags = {
            "phase": "final_training" if is_final_training else "tuning",
            "model_type": config["model_type"],
            "dataset_version": config["dataset_version"],
        }

        run_name = (
            "Final Model - SkipGram" if is_final_training else "Hyperparameter Tuning"
        )

        mlflow_run = mlflow.start_run(run_name=run_name, tags=tags)

        params_to_log = {}
        for k, v in config.items():
            if k in [
                "max_epochs",
                "batch_size",
                "num_negative_samples",
                "window_size",
                "num_workers",
                "learning_rate",
                "l2_reg",
                "embedding_dim",
            ]:
                params_to_log[f"{'final' if is_final_training else 'tuning'}.{k}"] = v
        mlflow.log_params(params_to_log)

        mlflow.log_params(
            {
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": (
                    torch.version.cuda if torch.cuda.is_available() else "N/A"
                ),
                "num_cpus": psutil.cpu_count(),
                "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            }
        )

    except Exception as e:
        print(f"Warning: Failed to initialize MLflow: {str(e)}")

    try:
        print("GPU available:", torch.cuda.is_available())
        data_path = config["data_path"]
        sequences_fp = os.path.join(data_path, config["sequences_file"])
        val_sequences_fp = os.path.join(data_path, config["val_sequences_file"])
        idm_fp = os.path.join(data_path, config["idm_file"])

        idm = IDMapper().load(idm_fp)
        dataset = SkipGramDataset(
            sequences_fp,
            window_size=config["window_size"],
            negative_samples=config["num_negative_samples"],
            id_to_idx=idm.item_to_index,
        )

        val_dataset = SkipGramDataset(
            val_sequences_fp,
            dataset.interacted,
            dataset.item_freq,
            window_size=config["window_size"],
            negative_samples=config["num_negative_samples"],
            id_to_idx=idm.item_to_index,
        )

        train_loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            drop_last=False,
            collate_fn=dataset.collate_fn,
            num_workers=config["num_workers"],
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            drop_last=False,
            collate_fn=val_dataset.collate_fn,
            num_workers=config["num_workers"],
        )

        model = SkipGram(dataset.vocab_size, config["embedding_dim"])
        lit_model = LitSkipGram(
            model,
            learning_rate=config["learning_rate"],
            l2_reg=config["l2_reg"],
            log_dir=os.path.join(
                config["checkpoint_dir"],
                (
                    f"trial_{train.get_context().get_trial_id()}"
                    if not is_final_training
                    else "final_model"
                ),
            ),
        )

        start_time = time.time()

        trainer = L.Trainer(
            max_epochs=config["max_epochs"],
            accelerator=config["accelerator"],
            callbacks=[
                L.pytorch.callbacks.ModelCheckpoint(
                    dirpath=os.path.join(
                        config["checkpoint_dir"],
                        (
                            f"trial_{train.get_context().get_trial_id()}"
                            if not is_final_training
                            else "final_model"
                        ),
                    ),
                    filename=config["checkpoint_filename"],
                    save_top_k=config["checkpoint_save_top_k"],
                    monitor=config["checkpoint_monitor"],
                    mode=config["checkpoint_mode"],
                ),
                EarlyStopping(
                    monitor=config["early_stopping_monitor"],
                    patience=config["early_stopping_patience"],
                    mode=config["early_stopping_mode"],
                    verbose=config["early_stopping_verbose"],
                ),
            ],
            logger=False,
        )

        trainer.fit(lit_model, train_loader, val_loader)

        training_time = time.time() - start_time

        val_loss = trainer.callback_metrics.get(
            "val_loss", torch.tensor(float("inf"))
        ).item()

        if not is_final_training:
            train.report({"val_loss": val_loss})

        if mlflow_run:
            try:
                metrics_to_log = {
                    "val_loss": val_loss,
                    "train_loss": trainer.callback_metrics.get(
                        "train_loss", torch.tensor(float("inf"))
                    ).item(),
                    "training_time_seconds": training_time,
                }

                if torch.cuda.is_available():
                    metrics_to_log["gpu_memory_allocated_mb"] = (
                        torch.cuda.memory_allocated() / (1024**2)
                    )
                    metrics_to_log["gpu_memory_reserved_mb"] = (
                        torch.cuda.memory_reserved() / (1024**2)
                    )

                metrics_to_log["cpu_memory_usage_percent"] = (
                    psutil.Process().memory_percent()
                )

                if is_final_training:
                    metrics_to_log = {
                        f"final.{k}": v for k, v in metrics_to_log.items()
                    }
                else:
                    metrics_to_log = {
                        f"tuning.{k}": v for k, v in metrics_to_log.items()
                    }

                for metric_name, metric_value in metrics_to_log.items():
                    try:
                        mlflow.log_metric(metric_name, metric_value)
                        print(f"Logged metric {metric_name}: {metric_value}")
                    except Exception as e:
                        print(f"Warning: Failed to log metric {metric_name}: {str(e)}")

                if is_final_training:
                    try:
                        idm_temp_path = os.path.join(
                            config["checkpoint_dir"], "id_mapper.json"
                        )
                        idm.save(idm_temp_path)
                        mlflow.log_artifact(idm_temp_path, artifact_path="id_mapper")
                        print(
                            f"IDMapper saved and logged as artifact at {idm_temp_path}"
                        )

                        import numpy as np
                        from mlflow.models.signature import ModelSignature
                        from mlflow.types.schema import Schema, TensorSpec

                        input_schema = Schema(
                            [
                                TensorSpec(
                                    name="target_items",
                                    type=np.dtype(np.int64),
                                    shape=(-1,),
                                ),
                                TensorSpec(
                                    name="context_items",
                                    type=np.dtype(np.int64),
                                    shape=(-1,),
                                ),
                            ]
                        )
                        output_schema = Schema(
                            [TensorSpec(type=np.dtype(np.float32), shape=(-1,))]
                        )
                        signature = ModelSignature(
                            inputs=input_schema, outputs=output_schema
                        )

                        model_metadata = {
                            "model_type": config["model_type"],
                            "task": "item-embedding",
                            "framework": "pytorch",
                            "description": """
                            SkipGram model for learning item embeddings from purchase sequences.
                            Model Details:
                            - Architecture: Single embedding layer with Xavier uniform initialization
                            - Input: Pairs of target and context item indices
                            - Output: Similarity score between items (0-1)
                            - Training: Negative sampling with frequency-based sampling
                            Typical Use Cases:
                            - Product recommendation
                            - Similar item finding
                            - Purchase sequence analysis
                            Input Format:
                            - target_items: Tensor of item indices (int64)
                            - context_items: Tensor of context item indices (int64)
                            Output Format:
                            - Tensor of similarity scores (float32)
                            Additional Artifacts:
                            - id_mapper/id_mapper.json: Mapping of item IDs to indices
                            """,
                            "hyperparameters": {
                                "embedding_dim": config["embedding_dim"],
                                "learning_rate": config["learning_rate"],
                                "l2_reg": config["l2_reg"],
                            },
                        }

                        try:
                            scripted_model = torch.jit.script(lit_model.skipgram_model)
                            mlflow.pytorch.log_model(
                                pytorch_model=scripted_model,
                                artifact_path="skipgram_model",
                                registered_model_name=f"{config['run_name']}_skipgram",
                                signature=signature,
                                metadata=model_metadata,
                            )
                            print("TorchScript model logged successfully.")
                        except Exception as e:
                            print(f"Failed to log TorchScript model: {str(e)}")

                        client = mlflow.tracking.MlflowClient()
                        latest_versions = client.get_latest_versions(
                            f"{config['run_name']}_skipgram", stages=["None"]
                        )
                        client.update_model_version(
                            name=f"{config['run_name']}_skipgram",
                            version=latest_versions[0].version,
                            description=model_metadata["description"],
                        )
                        this_version = latest_versions[0]
                        this_val_loss = val_loss

                        is_champion = True
                        worse_versions = []
                        for v in client.search_model_versions(
                            f"name='{config['run_name']}_skipgram'"
                        ):
                            if v.run_id == this_version.run_id:
                                continue
                            try:
                                run_data = client.get_run(v.run_id).data
                                other_val_loss = run_data.metrics.get(
                                    "final.val_loss"
                                ) or run_data.metrics.get("tuning.val_loss")
                                if (
                                    other_val_loss is not None
                                    and other_val_loss < this_val_loss
                                ):
                                    is_champion = False
                                    break
                                if other_val_loss is not None:
                                    worse_versions.append(v)
                            except Exception as e:
                                print(
                                    f"Warning: Cannot load run data for version {v.version}: {str(e)}"
                                )

                        if is_champion:
                            client.set_model_version_tag(
                                name=f"{config['run_name']}_skipgram",
                                version=this_version.version,
                                key="champion",
                                value="true",
                            )
                            print(
                                f"Model version {this_version.version} is now CHAMPION (val_loss = {this_val_loss:.4f})"
                            )
                            for v in worse_versions:
                                try:
                                    client.delete_model_version_tag(
                                        name=f"{config['run_name']}_skipgram",
                                        version=v.version,
                                        key="champion",
                                    )
                                    print(
                                        f"Removed champion tag from version {v.version}"
                                    )
                                except Exception as e:
                                    print(
                                        f"Warning: Failed to remove champion tag from version {v.version}: {str(e)}"
                                    )
                        else:
                            print(
                                f"Model version {this_version.version} is NOT champion (val_loss = {this_val_loss:.4f})"
                            )

                    except Exception as e:
                        print(f"Error logging model or IDMapper to MLflow: {str(e)}")

            except Exception as e:
                print(f"Warning: Failed to log metrics to MLflow: {str(e)}")

        print("Training completed, val_loss:", val_loss)

    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise
    finally:
        if mlflow_run:
            try:
                mlflow.end_run()
            except Exception:
                pass


if __name__ == "__main__":
    config = load_config("../../configs/item2vec.yaml")
    print("Loaded config:", config)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.chmod(config["checkpoint_dir"], 0o777)
    os.makedirs(config["storage_path"], exist_ok=True)
    os.chmod(config["storage_path"], 0o777)
    os.makedirs(config["final_checkpoint_dir"], exist_ok=True)
    os.chmod(config["final_checkpoint_dir"], 0o777)

    current_dir = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

    ray.init(
        address=config["ray"]["address"],
        runtime_env={
            "working_dir": config["ray"]["runtime_env"]["working_dir"],
            "py_modules": config["ray"]["runtime_env"]["py_modules"],
            "env_vars": {
                "PYTHONPATH": config["ray"]["runtime_env"]["env_vars"]["PYTHONPATH"],
                "MLFLOW_S3_ENDPOINT_URL": config["mlflow"]["s3_endpoint_url"],
                "AWS_ACCESS_KEY_ID": config["mlflow"]["aws_access_key_id"],
                "AWS_SECRET_ACCESS_KEY": config["mlflow"]["aws_secret_access_key"],
                "MLFLOW_TRACKING_URI": config["mlflow"]["tracking_uri"],
                "MLFLOW_S3_IGNORE_TLS": str(config["mlflow"]["s3_ignore_tls"]).lower(),
            },
        },
    )

    train_loop_config = {
        "data_path": config["data_path"],
        "sequences_file": config["sequences_file"],
        "val_sequences_file": config["val_sequences_file"],
        "idm_file": config["idm_file"],
        "checkpoint_dir": config["checkpoint_dir"],
        "run_name": config["experiment"]["run_name"],
        "dataset_version": config["experiment"]["dataset_version"],
        "model_type": config["model"]["type"],
        "mlflow_tracking_uri": config["mlflow"]["tracking_uri"],
        "max_epochs": config["training"]["max_epochs"],
        "batch_size": config["training"]["batch_size"],
        "num_negative_samples": config["training"]["num_negative_samples"],
        "window_size": config["training"]["window_size"],
        "num_workers": config["training"]["num_workers"],
        "accelerator": config["trainer"]["accelerator"],
        "checkpoint_filename": config["trainer"]["checkpoint"]["filename"],
        "checkpoint_save_top_k": config["trainer"]["checkpoint"]["save_top_k"],
        "checkpoint_monitor": config["trainer"]["checkpoint"]["monitor"],
        "checkpoint_mode": config["trainer"]["checkpoint"]["mode"],
        "early_stopping_monitor": config["trainer"]["early_stopping"]["monitor"],
        "early_stopping_patience": config["trainer"]["early_stopping"]["patience"],
        "early_stopping_mode": config["trainer"]["early_stopping"]["mode"],
        "early_stopping_verbose": config["trainer"]["early_stopping"]["verbose"],
    }

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=train_loop_config,
        scaling_config=ScalingConfig(num_workers=1, use_gpu=True),
    )

    param_space = {
        "train_loop_config": {
            "embedding_dim": tune.choice(config["model"]["embedding_dim"]["choice"]),
            "learning_rate": tune.loguniform(
                *config["training"]["learning_rate"]["loguniform"]
            ),
            "l2_reg": tune.loguniform(*config["training"]["l2_reg"]["loguniform"]),
        }
    }

    tune_config = tune.TuneConfig(
        metric=config["experiment"]["tune_config"]["metric"],
        mode=config["experiment"]["tune_config"]["mode"],
        num_samples=config["experiment"]["tune_config"]["num_samples"],
    )

    tuner = Tuner(
        trainer,
        param_space=param_space,
        tune_config=tune_config,
        run_config=RunConfig(
            storage_path=config["storage_path"],
            name=config["experiment"]["run_name"],
            checkpoint_config=CheckpointConfig(num_to_keep=1),
        ),
    )

    try:
        results = tuner.fit()
        print("Tuning completed, num trials:", len(results))
        if results:
            best_trial = results.get_best_result("val_loss", "min")
            print("Best trial config:", best_trial.config)
            print("Best trial val_loss:", best_trial.metrics["val_loss"])

            print("\nStarting final training with best parameters...")
            best_config = best_trial.config["train_loop_config"]
            final_config = {
                "data_path": config["data_path"],
                "sequences_file": config["sequences_file"],
                "val_sequences_file": config["val_sequences_file"],
                "idm_file": config["idm_file"],
                "checkpoint_dir": config["final_checkpoint_dir"],
                "run_name": config["experiment"]["run_name"],
                "dataset_version": config["experiment"]["dataset_version"],
                "model_type": config["model"]["type"],
                "mlflow_tracking_uri": config["mlflow"]["tracking_uri"],
                "max_epochs": config["training"]["max_epochs"],
                "batch_size": config["training"]["batch_size"],
                "num_negative_samples": config["training"]["num_negative_samples"],
                "window_size": config["training"]["window_size"],
                "num_workers": config["training"]["num_workers"],
                "embedding_dim": best_config["embedding_dim"],
                "learning_rate": best_config["learning_rate"],
                "l2_reg": best_config["l2_reg"],
                "accelerator": config["trainer"]["accelerator"],
                "checkpoint_filename": config["trainer"]["checkpoint"]["filename"],
                "checkpoint_save_top_k": config["trainer"]["checkpoint"]["save_top_k"],
                "checkpoint_monitor": config["trainer"]["checkpoint"]["monitor"],
                "checkpoint_mode": config["trainer"]["checkpoint"]["mode"],
                "early_stopping_monitor": config["trainer"]["early_stopping"][
                    "monitor"
                ],
                "early_stopping_patience": config["trainer"]["early_stopping"][
                    "patience"
                ],
                "early_stopping_mode": config["trainer"]["early_stopping"]["mode"],
                "early_stopping_verbose": config["trainer"]["early_stopping"][
                    "verbose"
                ],
            }

            final_trainer = TorchTrainer(
                train_loop_per_worker=lambda config: train_func(
                    config, is_final_training=True
                ),
                train_loop_config=final_config,
                scaling_config=ScalingConfig(num_workers=1, use_gpu=True),
            )

            final_result = final_trainer.fit()
            print("\nFinal training completed!")
            print("Final model checkpoint saved at:", config["final_checkpoint_dir"])

    except Exception as e:
        print("Tuning failed:", str(e))
        raise
