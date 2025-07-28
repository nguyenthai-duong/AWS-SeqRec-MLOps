from typing import NamedTuple

import kfp
from kfp import dsl
from kfp.components import func_to_container_op
from kubernetes.client import V1EnvVar, V1EnvVarSource, V1SecretKeySelector


def feature_engineering_op(
    output_path: str,
) -> NamedTuple("Outputs", [("output_path", str)]):
    import os
    import subprocess
    import sys
    from pathlib import Path

    # import uuid
    # output_path = f"{output_path}-{str(uuid.uuid4())[:8]}"
    # Debug thông tin về volume mounts
    print("Checking volume mounts:")
    print(f"Contents of /app: {os.listdir('/app')}")
    print(
        f"Contents of /app/src/feature_engineer: {os.listdir('/app/src/feature_engineer')}"
    )
    print(f"Contents of /data: {os.listdir('/data')}")

    # Tạo thư mục chứa output nếu chưa có
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Đảm bảo thư mục output tồn tại
    os.makedirs("/tmp/outputs/output_path", exist_ok=True)

    # Kiểm tra sự tồn tại của file
    pipeline_path = "/app/src/feature_engineer/000_feature_pipeline.py"
    working_dir = "/app/src/feature_engineer"
    print(f"Looking for pipeline at: {pipeline_path}")
    print(f"Current working directory: {os.getcwd()}")
    print(
        f"Directory contents of /app/src/feature_engineer: {os.listdir('/app/src/feature_engineer')}"
    )

    if not os.path.exists(pipeline_path):
        print(f"Error: File not found at {pipeline_path}")
        print("Current working directory:", os.getcwd())
        print("Directory contents:", os.listdir("/app/src/feature_engineer"))
        sys.exit(1)

    try:
        print(f"Running command: uv run {pipeline_path}")
        subprocess.run(["uv", "run", pipeline_path], check=True, cwd=working_dir)
    except subprocess.CalledProcessError as e:
        print(f"Error running pipeline: {e}")
        print("Command output:", e.output if hasattr(e, "output") else "No output")
        sys.exit(1)

    return (output_path,)


def negative_sampling_op(
    output_path: str,
) -> NamedTuple("Outputs", [("output_path", str)]):
    import os
    import subprocess
    import sys
    from pathlib import Path

    print("Checking volume mounts:")
    print(f"Contents of /app: {os.listdir('/app')}")
    print(
        f"Contents of /app/src/feature_engineer: {os.listdir('/app/src/feature_engineer')}"
    )
    print(f"Contents of /data: {os.listdir('/data')}")

    # Tạo thư mục chứa output nếu chưa có
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Đảm bảo thư mục output tồn tại
    os.makedirs("/tmp/outputs/output_path", exist_ok=True)

    # Kiểm tra sự tồn tại của file
    pipeline_path = "/app/src/feature_engineer/010_negative_sample.py"
    working_dir = "/app/src/feature_engineer"
    print(f"Looking for pipeline at: {pipeline_path}")
    print(f"Current working directory: {os.getcwd()}")
    print(
        f"Directory contents of /app/src/feature_engineer: {os.listdir('/app/src/feature_engineer')}"
    )

    if not os.path.exists(pipeline_path):
        print(f"Error: File not found at {pipeline_path}")
        print("Current working directory:", os.getcwd())
        print("Directory contents:", os.listdir("/app/src/feature_engineer"))
        sys.exit(1)

    try:
        print(f"Running command: uv run {pipeline_path}")
        subprocess.run(["uv", "run", pipeline_path], check=True, cwd=working_dir)
    except subprocess.CalledProcessError as e:
        print(f"Error running pipeline: {e}")
        print("Command output:", e.output if hasattr(e, "output") else "No output")
        sys.exit(1)

    return (output_path,)


def prep_item2vec_op(output_path: str) -> NamedTuple("Outputs", [("output_path", str)]):
    import os
    import subprocess
    import sys
    from pathlib import Path

    # Debug thông tin về volume mounts
    print("Checking volume mounts:")
    print(f"Contents of /app: {os.listdir('/app')}")
    print(
        f"Contents of /app/src/feature_engineer: {os.listdir('/app/src/feature_engineer')}"
    )
    print(f"Contents of /data: {os.listdir('/data')}")

    # Tạo thư mục chứa output nếu chưa có
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Đảm bảo thư mục output tồn tại
    os.makedirs("/tmp/outputs/output_path", exist_ok=True)

    # Kiểm tra sự tồn tại của file
    pipeline_path = "/app/src/feature_engineer/020_prep_item2vec.py"
    working_dir = "/app/src/feature_engineer"
    print(f"Looking for pipeline at: {pipeline_path}")
    print(f"Current working directory: {os.getcwd()}")
    print(
        f"Directory contents of /app/src/feature_engineer: {os.listdir('/app/src/feature_engineer')}"
    )

    if not os.path.exists(pipeline_path):
        print(f"Error: File not found at {pipeline_path}")
        print("Current working directory:", os.getcwd())
        print("Directory contents:", os.listdir("/app/src/feature_engineer"))
        sys.exit(1)

    # Chạy 02_prep_item2vec.py bằng uv run
    try:
        print(f"Running command: uv run {pipeline_path}")
        subprocess.run(["uv", "run", pipeline_path], check=True, cwd=working_dir)
    except subprocess.CalledProcessError as e:
        print(f"Error running pipeline: {e}")
        print("Command output:", e.output if hasattr(e, "output") else "No output")
        sys.exit(1)

    return (output_path,)


def train_item2vec_op(
    output_path: str,
) -> NamedTuple("Outputs", [("output_path", str)]):
    import os
    import subprocess
    import sys
    from pathlib import Path

    # Debug thông tin về volume mounts
    print("Checking volume mounts:")
    print(f"Contents of /app: {os.listdir('/app')}")
    print(
        f"Contents of /app/src/model_item2vec: {os.listdir('/app/src/model_item2vec')}"
    )
    print(f"Contents of /data: {os.listdir('/data')}")

    # Tạo thư mục chứa output nếu chưa có
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Đảm bảo thư mục output tồn tại
    os.makedirs("/tmp/outputs/output_path", exist_ok=True)

    # Kiểm tra sự tồn tại của file main.py
    script_path = "/app/src/model_item2vec/main.py"
    working_dir = "/app/src/model_item2vec"
    print(f"Looking for script at: {script_path}")
    print(f"Current working directory: {os.getcwd()}")
    print(
        f"Directory contents of /app/src/model_item2vec: {os.listdir('/app/src/model_item2vec')}"
    )

    if not os.path.exists(script_path):
        print(f"Error: File not found at {script_path}")
        print("Current working directory:", os.getcwd())
        print("Directory contents:", os.listdir("/app/src/model_item2vec"))
        sys.exit(1)

    # Chạy main.py bằng uv run
    try:
        print(f"Running command: uv run {script_path}")
        env = os.environ.copy()
        env["PYTHONPATH"] = "/app/src"
        subprocess.run(["uv", "run", script_path], check=True, cwd=working_dir, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")
        print("Command output:", e.output if hasattr(e, "output") else "No output")
        sys.exit(1)

    return (output_path,)


def train_ranking_sequence_op(
    output_path: str,
) -> NamedTuple("Outputs", [("output_path", str)]):
    import os
    import subprocess
    import sys
    from pathlib import Path

    # Debug thông tin về volume mounts
    print("Checking volume mounts:")
    print(f"Contents of /app: {os.listdir('/app')}")
    print(
        f"Contents of /app/src/model_ranking_sequence: {os.listdir('/app/src/model_ranking_sequence')}"
    )
    print(f"Contents of /data: {os.listdir('/data')}")

    # Tạo thư mục chứa output nếu chưa có
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Đảm bảo thư mục output tồn tại
    os.makedirs("/tmp/outputs/output_path", exist_ok=True)

    # Kiểm tra sự tồn tại của file main.py
    script_path = "/app/src/model_ranking_sequence/main.py"
    working_dir = "/app/src/model_ranking_sequence"
    print(f"Looking for script at: {script_path}")
    print(f"Current working directory: {os.getcwd()}")
    print(
        f"Directory contents of /app/src/model_ranking_sequence: {os.listdir('/app/src/model_ranking_sequence')}"
    )

    if not os.path.exists(script_path):
        print(f"Error: File not found at {script_path}")
        print("Current working directory:", os.getcwd())
        print("Directory contents:", os.listdir("/app/src/model_ranking_sequence"))
        sys.exit(1)

    # Chạy main.py bằng uv run
    try:
        print(f"Running command: uv run {script_path}")
        env = os.environ.copy()
        env["PYTHONPATH"] = "/app/src"
        subprocess.run(["uv", "run", script_path], check=True, cwd=working_dir, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")
        print("Command output:", e.output if hasattr(e, "output") else "No output")
        sys.exit(1)

    return (output_path,)


# Tạo component từ các hàm trên
feature_component = func_to_container_op(
    feature_engineering_op,
    base_image="kubeflow-pipeline:v5",
)

negative_sampling_component = func_to_container_op(
    negative_sampling_op,
    base_image="kubeflow-pipeline:v5",
)

prep_item2vec_component = func_to_container_op(
    prep_item2vec_op,
    base_image="kubeflow-pipeline:v5",
)

train_item2vec_component = func_to_container_op(
    train_item2vec_op,
    base_image="kubeflow-pipeline:v5",
)

train_ranking_sequence_component = func_to_container_op(
    train_ranking_sequence_op,
    base_image="kubeflow-pipeline:v5",
)


# Định nghĩa pipeline
@dsl.pipeline(
    name="Feature Engineering Pipeline",
    description="Pipeline for running feature engineering, negative sampling and item2vec preparation",
)
def feature_pipeline():
    # Use dsl.PipelineVolume to reference the existing PVC directly
    pvc = dsl.PipelineVolume(pvc="data-pvc")

    feature_task = (
        feature_component(output_path="/data/papermill-output/feature-output.ipynb")
        .add_pvolumes({"/data": pvc})
        .add_env_variable(V1EnvVar(name="PVC_PATH", value="/data"))
        .add_env_variable(
            V1EnvVar(
                name="AWS_REGION",
                value_from=V1EnvVarSource(
                    secret_key_ref=V1SecretKeySelector(
                        name="aws-credentials", key="AWS_REGION"
                    )
                ),
            )
        )
        .add_env_variable(
            V1EnvVar(
                name="AWS_ACCESS_KEY_ID",
                value_from=V1EnvVarSource(
                    secret_key_ref=V1SecretKeySelector(
                        name="aws-credentials", key="AWS_ACCESS_KEY_ID"
                    )
                ),
            )
        )
        .add_env_variable(
            V1EnvVar(
                name="AWS_SECRET_ACCESS_KEY",
                value_from=V1EnvVarSource(
                    secret_key_ref=V1SecretKeySelector(
                        name="aws-credentials", key="AWS_SECRET_ACCESS_KEY"
                    )
                ),
            )
        )
        .add_env_variable(
            V1EnvVar(
                name="S3_BUCKET",
                value_from=V1EnvVarSource(
                    secret_key_ref=V1SecretKeySelector(
                        name="aws-credentials", key="S3_BUCKET"
                    )
                ),
            )
        )
        .add_env_variable(
            V1EnvVar(
                name="POSTGRES_URI_REGISTRY",
                value_from=V1EnvVarSource(
                    secret_key_ref=V1SecretKeySelector(
                        name="aws-credentials", key="POSTGRES_URI_REGISTRY"
                    )
                ),
            )
        )
        .add_pod_annotation("debug/mount-path", "/data")
        .set_memory_request("2Gi")
    )

    feature_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

    negative_sampling_task = (
        negative_sampling_component(
            output_path="/data/papermill-output/negative-sampling-output.ipynb"
        )
        .add_pvolumes({"/data": pvc})
        .add_env_variable(V1EnvVar(name="PVC_PATH", value="/data"))
        .add_env_variable(
            V1EnvVar(
                name="AWS_REGION",
                value_from=V1EnvVarSource(
                    secret_key_ref=V1SecretKeySelector(
                        name="aws-credentials", key="AWS_REGION"
                    )
                ),
            )
        )
        .add_env_variable(
            V1EnvVar(
                name="AWS_ACCESS_KEY_ID",
                value_from=V1EnvVarSource(
                    secret_key_ref=V1SecretKeySelector(
                        name="aws-credentials", key="AWS_ACCESS_KEY_ID"
                    )
                ),
            )
        )
        .add_env_variable(
            V1EnvVar(
                name="AWS_SECRET_ACCESS_KEY",
                value_from=V1EnvVarSource(
                    secret_key_ref=V1SecretKeySelector(
                        name="aws-credentials", key="AWS_SECRET_ACCESS_KEY"
                    )
                ),
            )
        )
        .add_env_variable(
            V1EnvVar(
                name="S3_BUCKET",
                value_from=V1EnvVarSource(
                    secret_key_ref=V1SecretKeySelector(
                        name="aws-credentials", key="S3_BUCKET"
                    )
                ),
            )
        )
        .add_env_variable(
            V1EnvVar(
                name="POSTGRES_URI_REGISTRY",
                value_from=V1EnvVarSource(
                    secret_key_ref=V1SecretKeySelector(
                        name="aws-credentials", key="POSTGRES_URI_REGISTRY"
                    )
                ),
            )
        )
        .add_pod_annotation("debug/mount-path", "/data")
        .set_memory_request("2Gi")
        .after(feature_task)
    )

    negative_sampling_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

    prep_item2vec_task = (
        prep_item2vec_component(
            output_path="/data/papermill-output/prep-item2vec-output.ipynb"
        )
        .add_pvolumes({"/data": pvc})
        .add_env_variable(V1EnvVar(name="PVC_PATH", value="/data"))
        .add_pod_annotation("debug/mount-path", "/data")
        .set_memory_request("2Gi")
        .after(feature_task)
    )
    prep_item2vec_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

    # Add the new train_item2vec_task
    train_item2vec_task = (
        train_item2vec_component(
            output_path="/data/papermill-output/train-item2vec-output"
        )
        .add_pvolumes({"/data": pvc})
        .add_env_variable(V1EnvVar(name="PVC_PATH", value="/data"))
        .add_pod_annotation("debug/mount-path", "/data")
        .set_memory_request("2Gi")
        .after(prep_item2vec_task)
    )

    train_item2vec_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

    # Add the new train_ranking_sequence_task
    train_ranking_sequence_task = (
        train_ranking_sequence_component(
            output_path="/data/papermill-output/train-ranking_sequence-output"
        )
        .add_pvolumes({"/data": pvc})
        .add_env_variable(V1EnvVar(name="PVC_PATH", value="/data"))
        .add_pod_annotation("debug/mount-path", "/data")
        .set_memory_request("2Gi")
        .after(train_item2vec_task, negative_sampling_task)
    )

    train_ranking_sequence_task.execution_options.caching_strategy.max_cache_staleness = (
        "P0D"
    )


# Compile pipeline thành file YAML
if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=feature_pipeline, package_path="feature_pipeline.yaml"
    )
