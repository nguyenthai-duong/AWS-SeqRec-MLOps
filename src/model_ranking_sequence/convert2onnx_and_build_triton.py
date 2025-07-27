#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Best practice script: Convert PyTorch model to ONNX with ensemble preprocessing for Triton deployment
Supports single-item testing with corrected id_mapper to fix batch size mismatch
"""
import logging
import os
import shutil
import sys

import boto3
import dill
import mlflow
import pandas as pd
import torch
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking import MlflowClient

try:
    import onnx
    from onnx import checker
except ImportError:
    raise ImportError(
        "Please install the 'onnx' package to run this script: pip install onnx"
    )

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# MinIO and MLflow setup
AWS_KEY = "admin"
AWS_SECRET = "Password1234"
MINIO_EP = "http://minio-service.mlflow.svc.cluster.local:9000"
MLFLOW_URI = "http://mlflow-tracking-service.mlflow.svc.cluster.local:5000"
MODEL_NAME = os.getenv("MODEL_NAME", "seq_tune_v1_sequence_rating")
TRITON_REPO = "./src/model_ranking_sequence/model_repository"

os.environ["AWS_ACCESS_KEY_ID"] = AWS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_EP
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

# Input configuration
SEQ_LEN = 10
FEATURE_SIZE = None  # Will be set dynamically
ONNX_OPSET = 13
MAX_USER_ID_LEN = -1  # Variable-length user IDs
MAX_ITEM_ID_LEN = 1  # Single item testing
MAX_CATEGORIES_LEN = -1

# Dynamic axes for ONNX
DYNAMIC_AXES = {
    "user_ids": {0: "batch_size"},
    "input_seq": {0: "batch_size", 1: "seq_len"},
    "input_seq_ts_bucket": {0: "batch_size", 1: "seq_len"},
    "item_features": {0: "batch_size"},
    "target_item": {0: "batch_size"},
    "output": {0: "batch_size"},
}


# S3/MinIO connection testing
def test_s3_connection():
    logger.info("⏳ Testing MinIO connection...")
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_EP,
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
    )
    try:
        buckets = s3.list_buckets()
        logger.info("✅ Buckets: %s", [b["Name"] for b in buckets["Buckets"]])
    except Exception as e:
        logger.error("❌ Error connecting to MinIO: %s", e)
        raise


boto_session = boto3.session.Session(
    aws_access_key_id=AWS_KEY,
    aws_secret_access_key=AWS_SECRET,
)


def patched_get_s3_client(self):
    return boto_session.client("s3", endpoint_url=MINIO_EP)


S3ArtifactRepository._get_s3_client = patched_get_s3_client
logger.info(
    "S3 Client Endpoint: %s",
    boto_session.client("s3", endpoint_url=MINIO_EP).meta.endpoint_url,
)


# Wrapper to adjust shapes for Triton export
class TritonModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self, user_ids, input_seq, input_seq_ts_bucket, item_features, target_item
    ):
        user_ids = user_ids.squeeze(1)
        target_item = target_item.squeeze(1)
        return self.model(
            user_ids, input_seq, input_seq_ts_bucket, item_features, target_item
        )


def find_champion_version(model_name: str) -> str:
    client = MlflowClient()
    versions = client.search_model_versions(f"name = '{model_name}'")
    for v in versions:
        if v.tags.get("champion", "").lower() == "true":
            logger.info("Found champion version: %s", v.version)
            return v.version
    if versions:
        latest = max(versions, key=lambda v: int(v.version))
        logger.info("No champion found, using latest version: %s", latest.version)
        return latest.version
    logger.error("No model versions found for %s", model_name)
    raise ValueError(f"No model versions found for {model_name}")


def validate_pipeline_feature_size(pipeline, expected_size: int = 416) -> int:
    """Validate the feature size output by the item metadata pipeline."""
    logger.info("⏳ Validating item metadata pipeline feature size...")
    dummy_data = pd.DataFrame(
        {
            "parent_asin": ["B00MVV114A"],
            "categories": ["Toys & Games|Games & Accessories|Board Games"],
            "price": [13.64],
            "main_category": ["Toys & Games"],
            "parent_asin_rating_cnt_365d": [1],
            "parent_asin_rating_avg_prev_rating_365d": [5.0],
            "parent_asin_rating_cnt_90d": [0],
            "parent_asin_rating_avg_prev_rating_90d": [0.0],
            "parent_asin_rating_cnt_30d": [0],
            "parent_asin_rating_avg_prev_rating_30d": [0.0],
            "parent_asin_rating_cnt_7d": [0],
            "parent_asin_rating_avg_prev_rating_7d": [0.0],
        }
    )
    features = pipeline.transform(dummy_data)
    feature_size = features.shape[1]
    logger.info("Item metadata pipeline feature size: %d", feature_size)
    if feature_size != expected_size:
        logger.warning(
            "Feature size mismatch: expected %d, got %d", expected_size, feature_size
        )
    return feature_size


def load_model_and_artifacts(model_name: str):
    client = MlflowClient()
    version = find_champion_version(model_name)
    model_uri = f"models:/{model_name}/{version}"
    logger.info("🔗 Loading model from: %s", model_uri)

    try:
        model = mlflow.pytorch.load_model(model_uri)
        model.eval()
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error("❌ Error loading model: %s", e)
        raise

    model_version = client.get_model_version(model_name, version)
    run_id = model_version.run_id
    logger.info("Run ID for model version %s: %s", version, run_id)

    # Create model repository directories
    id_mapper_dir = os.path.join(TRITON_REPO, "id_mapper", "1")
    item_pipeline_dir = os.path.join(TRITON_REPO, "item_pipeline", "1")
    os.makedirs(id_mapper_dir, exist_ok=True)
    os.makedirs(item_pipeline_dir, exist_ok=True)

    # Download and copy IDMapper
    try:
        idm_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="id_mapper/id_mapper.pkl"
        )
        if not os.path.isfile(idm_path):
            raise FileNotFoundError(f"IDMapper not found at {idm_path}")
        shutil.copy2(idm_path, os.path.join(id_mapper_dir, "id_mapper.pkl"))

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, base_dir)

        with open(idm_path, "rb") as f:
            idm = dill.load(f)
        if idm is None or not hasattr(idm, "get_user_index"):
            raise ValueError("Invalid IDMapper loaded")
        logger.info("✅ IDMapper loaded and copied to %s", id_mapper_dir)
    except Exception as e:
        logger.error("❌ Failed to load IDMapper: %s", e)
        raise

    # Download and copy item_metadata_pipeline
    try:
        pipeline_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="item_metadata_pipeline/item_metadata_pipeline.pkl",
        )
        if not os.path.isfile(pipeline_path):
            raise FileNotFoundError(
                f"Item metadata pipeline not found at {pipeline_path}"
            )
        shutil.copy2(
            pipeline_path, os.path.join(item_pipeline_dir, "item_metadata_pipeline.pkl")
        )
        with open(pipeline_path, "rb") as f:
            item_pipeline = dill.load(f)
        if item_pipeline is None:
            raise ValueError("Invalid item metadata pipeline loaded")
        logger.info(
            "✅ Item metadata pipeline loaded and copied to %s", item_pipeline_dir
        )
    except Exception as e:
        logger.error("❌ Failed to load item metadata pipeline: %s", e)
        raise

    # Compute FEATURE_SIZE dynamically
    global FEATURE_SIZE
    FEATURE_SIZE = validate_pipeline_feature_size(item_pipeline, expected_size=416)
    logger.info("Set FEATURE_SIZE to %d based on item pipeline output", FEATURE_SIZE)

    return model, idm, item_pipeline


def print_model_info(model: torch.nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("\n===== Model Summary =====")
    logger.info("%s", model)
    logger.info("Total parameters: %s", f"{total_params:,}")
    logger.info("Trainable parameters: %s", f"{trainable:,}")


def export_to_onnx(model: torch.nn.Module, seq_len: int, feature_size: int) -> str:
    logger.info("\n⏳ Exporting model to ONNX format...")
    wrapper = TritonModelWrapper(model)
    dummy_inputs = (
        torch.randint(0, 10, (1, 1), dtype=torch.int64),  # user_ids
        torch.randint(0, 10, (1, seq_len), dtype=torch.int64),  # input_seq
        torch.randint(
            0, seq_len + 1, (1, seq_len), dtype=torch.int64
        ),  # input_seq_ts_bucket
        torch.randn(1, feature_size, dtype=torch.float32),  # item_features
        torch.randint(0, 10, (1, 1), dtype=torch.int64),  # target_item
    )
    onnx_path = os.path.join(TRITON_REPO, "ranker", "1", "model.onnx")
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    try:
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            onnx_path,
            export_params=True,
            opset_version=ONNX_OPSET,
            do_constant_folding=True,
            input_names=[
                "user_ids",
                "input_seq",
                "input_seq_ts_bucket",
                "item_features",
                "target_item",
            ],
            output_names=["output"],
            dynamic_axes=DYNAMIC_AXES,
            verbose=False,
        )
        logger.info("✅ ONNX model exported: %s", onnx_path)
        onnx_model = onnx.load(onnx_path)
        checker.check_model(onnx_model)
        logger.info("✅ ONNX model validation passed")
        return onnx_path
    except Exception as e:
        logger.error("❌ ONNX export/check failed: %s", e)
        raise


def prepare_triton_repo(onnx_path: str, model_name: str, repo_path: str):
    # Verify ONNX file exists
    if not os.path.isfile(onnx_path):
        logger.error("❌ ONNX file does not exist: %s", onnx_path)
        raise FileNotFoundError(f"ONNX file does not exist: {onnx_path}")

    # Create model directories
    ranker_dir = os.path.join(repo_path, "ranker")
    id_mapper_dir = os.path.join(repo_path, "id_mapper")
    item_pipeline_dir = os.path.join(repo_path, "item_pipeline")
    ensemble_dir = os.path.join(repo_path, "ensemble")

    for model_dir in [ranker_dir, id_mapper_dir, item_pipeline_dir, ensemble_dir]:
        version_dir = os.path.join(model_dir, "1")
        os.makedirs(version_dir, exist_ok=True)
        logger.info("✅ Created directory: %s", version_dir)

    # Verify pickle files
    id_mapper_pkl_path = os.path.join(id_mapper_dir, "1", "id_mapper.pkl")
    item_pipeline_pkl_path = os.path.join(
        item_pipeline_dir, "1", "item_metadata_pipeline.pkl"
    )
    for pkl_path in [id_mapper_pkl_path, item_pipeline_pkl_path]:
        if not os.path.isfile(pkl_path):
            logger.error("❌ Pickle file not found: %s", pkl_path)
            raise FileNotFoundError(f"Pickle file not found: {pkl_path}")
        logger.info("✅ Verified pickle file: %s", pkl_path)

    # Write ranker config
    ranker_config = f"""
name: "ranker"
platform: "onnxruntime_onnx"
max_batch_size: 512
input [
  {{ name: "user_ids", data_type: TYPE_INT64, dims: [1] }},
  {{ name: "input_seq", data_type: TYPE_INT64, dims: [{SEQ_LEN}] }},
  {{ name: "input_seq_ts_bucket", data_type: TYPE_INT64, dims: [{SEQ_LEN}] }},
  {{ name: "item_features", data_type: TYPE_FP32, dims: [{FEATURE_SIZE}] }},
  {{ name: "target_item", data_type: TYPE_INT64, dims: [1] }}
]
output [
  {{ name: "output", data_type: TYPE_FP32, dims: [1] }}
]
instance_group [ {{ kind: KIND_GPU, count: 4 }} ]
"""
    ranker_config_path = os.path.join(ranker_dir, "config.pbtxt")
    with open(ranker_config_path, "w", encoding="utf-8") as f:
        f.write(ranker_config)
    logger.info("✅ Ranker config created at %s", ranker_config_path)

    # Write IDMapper model script
    id_mapper_script = """
import dill
import numpy as np
import logging
import triton_python_backend_utils as pb_utils
import os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TritonPythonModel:
    def __init__(self):
        self.idm = None
        self.sequence_length = 10  # Match SEQ_LEN from script
        self.padding_value = -1   # Match padding_value from convert_asin_to_idx

    def initialize(self, args):
        model_dir = os.path.dirname(__file__)
        pkl_path = os.path.join(model_dir, "id_mapper.pkl")
        logger.info(f"Loading IDMapper from: {pkl_path}")
        with open(pkl_path, "rb") as f:
            self.idm = dill.load(f)
        logger.info("IDMapper initialized")

    def convert_to_idx(self, sequence, sequence_length, padding_value):
        # Handle None or empty sequences
        if sequence is None or len(sequence) == 0:
            return np.array([padding_value] * sequence_length, dtype=np.int64)

        # Convert items to indices
        indices = []
        for item in sequence:
            if item == -1:  # Preserve padding_value
                indices.append(padding_value)
            else:
                try:
                    idx = self.idm.get_item_index(str(item))
                    indices.append(idx)
                except (KeyError, ValueError):
                    indices.append(padding_value)  # Handle invalid items

        # Pad sequence if needed
        padding_needed = sequence_length - len(indices)
        if padding_needed > 0:
            indices = [padding_value] * padding_needed + indices
        elif padding_needed < 0:
            indices = indices[-sequence_length:]  # Truncate if too long

        return np.array(indices, dtype=np.int64)

    def execute(self, requests):
        responses = []
        for request in requests:
            user_ids = pb_utils.get_input_tensor_by_name(request, "user_ids").as_numpy()  # Shape: [batch_size, -1]
            target_items = pb_utils.get_input_tensor_by_name(request, "target_items").as_numpy()  # Shape: [batch_size, 1]
            input_seq = pb_utils.get_input_tensor_by_name(request, "input_seq").as_numpy()  # Shape: [batch_size, 10]

            batch_size = user_ids.shape[0]
            logger.info("Batch size: %d", batch_size)

            # Process user_ids: Take the first non-empty string per batch item
            user_indices = np.array([
                self.idm.get_user_index(uid[0].decode('utf-8')) for uid in user_ids
            ], dtype=np.int64).reshape(batch_size, 1)  # Shape: [batch_size, 1]

            # Process target_items
            target_indices = np.array([
                self.idm.get_item_index(item[0].decode('utf-8')) for item in target_items
            ], dtype=np.int64).reshape(batch_size, 1)  # Shape: [batch_size, 1]

            # Process input_seq with padding
            seq_indices = np.array([
                self.convert_to_idx(seq, self.sequence_length, self.padding_value)
                for seq in input_seq
            ], dtype=np.int64)  # Shape: [batch_size, 10]

            logger.info("user_indices shape: %s", user_indices.shape)
            logger.info("seq_indices shape: %s", seq_indices.shape)
            logger.info("target_indices shape: %s", target_indices.shape)

            user_indices_tensor = pb_utils.Tensor("user_indices", user_indices)
            seq_indices_tensor = pb_utils.Tensor("seq_indices", seq_indices)
            target_indices_tensor = pb_utils.Tensor("target_indices", target_indices)

            response = pb_utils.InferenceResponse(output_tensors=[
                user_indices_tensor,
                seq_indices_tensor,
                target_indices_tensor
            ])
            responses.append(response)
        return responses

    def finalize(self):
        logger.info("IDMapper finalized")
"""
    id_mapper_model_path = os.path.join(id_mapper_dir, "1", "model.py")
    with open(id_mapper_model_path, "w", encoding="utf-8") as f:
        f.write(id_mapper_script)
    logger.info("✅ IDMapper model script created at %s", id_mapper_model_path)

    # Write IDMapper config
    id_mapper_config = f"""
name: "id_mapper"
backend: "python"
max_batch_size: 512
input [
  {{ name: "user_ids", data_type: TYPE_STRING, dims: [{MAX_USER_ID_LEN}] }},
  {{ name: "input_seq", data_type: TYPE_STRING, dims: [-1] }},
  {{ name: "target_items", data_type: TYPE_STRING, dims: [{MAX_ITEM_ID_LEN}] }}
]
output [
  {{ name: "user_indices", data_type: TYPE_INT64, dims: [1] }},
  {{ name: "seq_indices", data_type: TYPE_INT64, dims: [{SEQ_LEN}] }},
  {{ name: "target_indices", data_type: TYPE_INT64, dims: [{MAX_ITEM_ID_LEN}] }}
]
instance_group [ {{ kind: KIND_CPU, count: 4 }} ]
"""
    id_mapper_config_path = os.path.join(id_mapper_dir, "config.pbtxt")
    with open(id_mapper_config_path, "w", encoding="utf-8") as f:
        f.write(id_mapper_config)
    logger.info("✅ IDMapper config created at %s", id_mapper_config_path)

    # Write item pipeline model script
    item_pipeline_script = """
import dill
import numpy as np
import pandas as pd
import logging
import triton_python_backend_utils as pb_utils
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TritonPythonModel:
    def __init__(self):
        self.pipeline = None

    def initialize(self, args):
        model_dir = os.path.dirname(__file__)
        pkl_path = os.path.join(model_dir, "item_metadata_pipeline.pkl")
        logger.info(f"Loading item pipeline from: {pkl_path}")
        with open(pkl_path, "rb") as f:
            self.pipeline = dill.load(f)
        logger.info("Item pipeline initialized")

    def execute(self, requests):
        responses = []
        for request in requests:
            items = pb_utils.get_input_tensor_by_name(request, "items").as_numpy()
            categories = pb_utils.get_input_tensor_by_name(request, "categories").as_numpy()
            prices = pb_utils.get_input_tensor_by_name(request, "prices").as_numpy()
            main_categories = pb_utils.get_input_tensor_by_name(request, "main_categories").as_numpy()
            rating_cnt_365d = pb_utils.get_input_tensor_by_name(request, "rating_cnt_365d").as_numpy()
            rating_avg_365d = pb_utils.get_input_tensor_by_name(request, "rating_avg_365d").as_numpy()
            rating_cnt_90d = pb_utils.get_input_tensor_by_name(request, "rating_cnt_90d").as_numpy()
            rating_avg_90d = pb_utils.get_input_tensor_by_name(request, "rating_avg_90d").as_numpy()
            rating_cnt_30d = pb_utils.get_input_tensor_by_name(request, "rating_cnt_30d").as_numpy()
            rating_avg_30d = pb_utils.get_input_tensor_by_name(request, "rating_avg_30d").as_numpy()
            rating_cnt_7d = pb_utils.get_input_tensor_by_name(request, "rating_cnt_7d").as_numpy()
            rating_avg_7d = pb_utils.get_input_tensor_by_name(request, "rating_avg_7d").as_numpy()

            batch_size = items.shape[0]
            logger.info(f"Batch size: {batch_size}")
            logger.info(f"items shape: {items.shape}")
            logger.info(f"categories shape: {categories.shape}")
            logger.info(f"prices shape: {prices.shape}")

            df_data = {
                "parent_asin": [item.decode('utf-8') for item in items.reshape(batch_size, -1)[:, 0]],
                "categories": [cat.decode('utf-8').replace('__', '|') for cat in categories.reshape(batch_size, -1)[:, 0]],
                "price": prices.reshape(batch_size, -1)[:, 0].astype(np.float32),
                "main_category": [mc.decode('utf-8') for mc in main_categories.reshape(batch_size, -1)[:, 0]],
                "parent_asin_rating_cnt_365d": rating_cnt_365d.reshape(batch_size, -1)[:, 0].astype(np.int64),
                "parent_asin_rating_avg_prev_rating_365d": rating_avg_365d.reshape(batch_size, -1)[:, 0].astype(np.float32),
                "parent_asin_rating_cnt_90d": rating_cnt_90d.reshape(batch_size, -1)[:, 0].astype(np.int64),
                "parent_asin_rating_avg_prev_rating_90d": rating_avg_90d.reshape(batch_size, -1)[:, 0].astype(np.float32),
                "parent_asin_rating_cnt_30d": rating_cnt_30d.reshape(batch_size, -1)[:, 0].astype(np.int64),
                "parent_asin_rating_avg_prev_rating_30d": rating_avg_30d.reshape(batch_size, -1)[:, 0].astype(np.float32),
                "parent_asin_rating_cnt_7d": rating_cnt_7d.reshape(batch_size, -1)[:, 0].astype(np.int64),
                "parent_asin_rating_avg_prev_rating_7d": rating_avg_7d.reshape(batch_size, -1)[:, 0].astype(np.float32)
            }

            for key, value in df_data.items():
                logger.info(f"DataFrame column {key} length: {len(value)}")

            df = pd.DataFrame(df_data)
            df.fillna(0.0, inplace=True)
            features = self.pipeline.transform(df).astype(np.float32)
            logger.info(f"item_features shape: {features.shape}")

            features_tensor = pb_utils.Tensor("item_features", features)
            response = pb_utils.InferenceResponse(output_tensors=[features_tensor])
            responses.append(response)
        return responses

    def finalize(self):
        logger.info("Item pipeline finalized")
"""
    item_pipeline_model_path = os.path.join(item_pipeline_dir, "1", "model.py")
    with open(item_pipeline_model_path, "w", encoding="utf-8") as f:
        f.write(item_pipeline_script)
    logger.info("✅ Item pipeline model script created at %s", item_pipeline_model_path)

    # Write item pipeline config
    item_pipeline_config = f"""
name: "item_pipeline"
backend: "python"
max_batch_size: 512
input [
  {{ name: "items", data_type: TYPE_STRING, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "categories", data_type: TYPE_STRING, dims: [{MAX_CATEGORIES_LEN}] }},
  {{ name: "prices", data_type: TYPE_FP32, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "main_categories", data_type: TYPE_STRING, dims: [{MAX_CATEGORIES_LEN}] }},
  {{ name: "rating_cnt_365d", data_type: TYPE_INT64, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "rating_avg_365d", data_type: TYPE_FP32, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "rating_cnt_90d", data_type: TYPE_INT64, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "rating_avg_90d", data_type: TYPE_FP32, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "rating_cnt_30d", data_type: TYPE_INT64, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "rating_avg_30d", data_type: TYPE_FP32, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "rating_cnt_7d", data_type: TYPE_INT64, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "rating_avg_7d", data_type: TYPE_FP32, dims: [{MAX_ITEM_ID_LEN}] }}
]
output [
  {{ name: "item_features", data_type: TYPE_FP32, dims: [{FEATURE_SIZE}] }}
]
instance_group [ {{ kind: KIND_CPU, count: 4 }} ]
"""
    item_pipeline_config_path = os.path.join(item_pipeline_dir, "config.pbtxt")
    with open(item_pipeline_config_path, "w", encoding="utf-8") as f:
        f.write(item_pipeline_config)
    logger.info("✅ Item pipeline config created at %s", item_pipeline_config_path)

    # Write ensemble config
    ensemble_config = f"""
name: "ensemble"
platform: "ensemble"
max_batch_size: 512
input [
  {{ name: "user_ids", data_type: TYPE_STRING, dims: [{MAX_USER_ID_LEN}] }},
  {{ name: "input_seq", data_type: TYPE_STRING, dims: [-1] }},
  {{ name: "input_seq_ts_bucket", data_type: TYPE_INT64, dims: [{SEQ_LEN}] }},
  {{ name: "items", data_type: TYPE_STRING, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "categories", data_type: TYPE_STRING, dims: [{MAX_CATEGORIES_LEN}] }},
  {{ name: "prices", data_type: TYPE_FP32, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "main_categories", data_type: TYPE_STRING, dims: [{MAX_CATEGORIES_LEN}] }},
  {{ name: "rating_cnt_365d", data_type: TYPE_INT64, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "rating_avg_365d", data_type: TYPE_FP32, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "rating_cnt_90d", data_type: TYPE_INT64, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "rating_avg_90d", data_type: TYPE_FP32, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "rating_cnt_30d", data_type: TYPE_INT64, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "rating_avg_30d", data_type: TYPE_FP32, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "rating_cnt_7d", data_type: TYPE_INT64, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "rating_avg_7d", data_type: TYPE_FP32, dims: [{MAX_ITEM_ID_LEN}] }}
]
output [
  {{ name: "output", data_type: TYPE_FP32, dims: [{MAX_ITEM_ID_LEN}] }}
]
ensemble_scheduling {{
  step [
    {{
      model_name: "id_mapper"
      model_version: -1
      input_map {{ key: "user_ids", value: "user_ids" }}
      input_map {{ key: "input_seq", value: "input_seq" }}
      input_map {{ key: "target_items", value: "items" }}
      output_map {{ key: "user_indices", value: "user_indices" }}
      output_map {{ key: "seq_indices", value: "seq_indices" }}
      output_map {{ key: "target_indices", value: "target_indices" }}
    }},
    {{
      model_name: "item_pipeline"
      model_version: -1
      input_map {{ key: "items", value: "items" }}
      input_map {{ key: "categories", value: "categories" }}
      input_map {{ key: "prices", value: "prices" }}
      input_map {{ key: "main_categories", value: "main_categories" }}
      input_map {{ key: "rating_cnt_365d", value: "rating_cnt_365d" }}
      input_map {{ key: "rating_avg_365d", value: "rating_avg_365d" }}
      input_map {{ key: "rating_cnt_90d", value: "rating_cnt_90d" }}
      input_map {{ key: "rating_avg_90d", value: "rating_avg_90d" }}
      input_map {{ key: "rating_cnt_30d", value: "rating_cnt_30d" }}
      input_map {{ key: "rating_avg_30d", value: "rating_avg_30d" }}
      input_map {{ key: "rating_cnt_7d", value: "rating_cnt_7d" }}
      input_map {{ key: "rating_avg_7d", value: "rating_avg_7d" }}
      output_map {{ key: "item_features", value: "item_features" }}
    }},
    {{
      model_name: "ranker"
      model_version: -1
      input_map {{ key: "user_ids", value: "user_indices" }}
      input_map {{ key: "input_seq", value: "seq_indices" }}
      input_map {{ key: "input_seq_ts_bucket", value: "input_seq_ts_bucket" }}
      input_map {{ key: "item_features", value: "item_features" }}
      input_map {{ key: "target_item", value: "target_indices" }}
      output_map {{ key: "output", value: "output" }}
    }}
  ]
}}
"""
    ensemble_config_path = os.path.join(ensemble_dir, "config.pbtxt")
    with open(ensemble_config_path, "w", encoding="utf-8") as f:
        f.write(ensemble_config)
    logger.info("✅ Ensemble config created at %s", ensemble_config_path)

    # === Copy id_mapper.py vào id_mapper/1/ ===
    IDM_SOURCE = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "id_mapper.py")
    )
    IDM_DEST = os.path.join(repo_path, "id_mapper", "1", "id_mapper.py")
    shutil.copy2(IDM_SOURCE, IDM_DEST)
    logger.info("✅ Copied id_mapper.py to %s", IDM_DEST)

    # === Copy thư mục features vào item_pipeline/1/features/ ===
    FEATURES_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "features"))
    FEATURES_DST = os.path.join(repo_path, "item_pipeline", "1", "features")
    if os.path.exists(FEATURES_DST):
        shutil.rmtree(FEATURES_DST)
    shutil.copytree(FEATURES_SRC, FEATURES_DST)
    logger.info("✅ Copied features/ to %s", FEATURES_DST)

    logger.info("🚀 Triton ensemble ready at: %s", os.path.abspath(repo_path))


def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    test_s3_connection()
    model, idm, item_pipeline = load_model_and_artifacts(MODEL_NAME)
    print_model_info(model)
    onnx_path = export_to_onnx(model, SEQ_LEN, FEATURE_SIZE)
    prepare_triton_repo(onnx_path, MODEL_NAME, TRITON_REPO)


if __name__ == "__main__":
    main()
