import os
import time
import requests
from mlflow.tracking import MlflowClient
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration from environment variables
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "seq_tune_v1_sequence_rating")
JENKINS_BASE = os.getenv("JENKINS_BASE", "http://localhost:8080/jenkins")  # Đổi thành endpoint có /jenkins nếu dùng context path
JENKINS_JOB = os.getenv("JENKINS_JOB", "pipeline_deploy_triton")
JENKINS_USER = os.getenv("JENKINS_USER", "admin")
JENKINS_TOKEN = os.getenv("JENKINS_TOKEN", "af49796b8a904968acdb85e8db32a7da")

JENKINS_URL = f"{JENKINS_BASE}/job/{JENKINS_JOB}/buildWithParameters"

# Initialize MLflow client
client = MlflowClient(tracking_uri=MLFLOW_URI)

def get_jenkins_crumb_and_cookies():
    """Lấy crumb và cookie từ Jenkins."""
    crumb_url = f"{JENKINS_BASE}/crumbIssuer/api/json"
    session = requests.Session()
    resp = session.get(crumb_url, auth=(JENKINS_USER, JENKINS_TOKEN), timeout=10)
    resp.raise_for_status()
    crumb_json = resp.json()
    crumb_field = crumb_json["crumbRequestField"]
    crumb = crumb_json["crumb"]
    # Cookie tự động được lưu trong session.cookies
    return crumb_field, crumb, session

def trigger_jenkins(model_name: str, version: str) -> bool:
    """Trigger Jenkins pipeline with model name and version (with crumb + cookie)."""
    try:
        crumb_field, crumb, session = get_jenkins_crumb_and_cookies()
        headers = {crumb_field: crumb}
        params = {"MODEL_NAME": model_name, "MODEL_VERSION": version}
        response = session.post(
            JENKINS_URL,
            auth=(JENKINS_USER, JENKINS_TOKEN),
            headers=headers,
            params=params,
            timeout=20,
        )
        ok = response.status_code in (200, 201, 202)
        logger.info(f"Trigger Jenkins for {model_name} v{version}: {response.status_code} - {response.text[:200]}")
        return ok
    except Exception as e:
        logger.error(f"Failed to trigger Jenkins for {model_name} v{version}: {e}")
        return False

def check_model_promotion():
    """Check for newly promoted champion models in MLflow and trigger Jenkins."""
    while True:
        try:
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            # Filter for production stage and champion tag
            prod_champion_versions = [
                v for v in versions
                if v.tags.get("stage", "").lower() == "production" and v.tags.get("champion", "").lower() == "true"
            ]

            if prod_champion_versions:
                # Sort by version number (descending) to get the latest
                prod_champion_versions.sort(key=lambda v: int(v.version), reverse=True)
                latest_version = prod_champion_versions[0]

                # Check if the model version has not been deployed yet
                if latest_version.tags.get("deploy", "").lower() != "true":
                    logger.info(f"Found promoted champion model {MODEL_NAME} v{latest_version.version}")
                    if trigger_jenkins(MODEL_NAME, latest_version.version):
                        client.set_model_version_tag(
                            MODEL_NAME, latest_version.version, "deploy", "true"
                        )
                        logger.info(f"Tagged {MODEL_NAME} v{latest_version.version} as deploy=true")
                    else:
                        logger.warning(f"Jenkins trigger failed for {MODEL_NAME} v{latest_version.version}")
                else:
                    logger.info(f"Model {MODEL_NAME} v{latest_version.version} already deployed")
            else:
                logger.info(f"No production champion models found for {MODEL_NAME}")

        except Exception as e:
            logger.error(f"Error in model promotion check: {e}")

        time.sleep(10)  # Poll every 10 seconds

if __name__ == "__main__":
    logger.info("Starting model promotion watcher...")
    check_model_promotion()
