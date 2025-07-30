import os
import time
import logging
import requests
from mlflow.tracking import MlflowClient


def configure_logging():
    """
    Configures the logging system with INFO level and a standardized format.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)


def load_configurations():
    """
    Loads configuration settings from environment variables.

    Returns:
        tuple: A tuple containing MLflow tracking URI, model name, Jenkins base URL,
               Jenkins job name, Jenkins user, and Jenkins token.
    """
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    model_name = os.getenv("MODEL_NAME", "seq_tune_v1_sequence_rating")
    jenkins_base = os.getenv("JENKINS_BASE", "http://localhost:8080/jenkins")
    jenkins_job = os.getenv("JENKINS_JOB", "pipeline_deploy_triton")
    jenkins_user = os.getenv("JENKINS_USER", "admin")
    jenkins_token = os.getenv("JENKINS_TOKEN", "af49796b8a904968acdb85e8db32a7da")
    return mlflow_uri, model_name, jenkins_base, jenkins_job, jenkins_user, jenkins_token


def initialize_mlflow_client(mlflow_uri):
    """
    Initializes the MLflow client with the specified tracking URI.

    Args:
        mlflow_uri (str): URI for the MLflow tracking server.

    Returns:
        MlflowClient: Initialized MLflow client instance.
    """
    return MlflowClient(tracking_uri=mlflow_uri)


logger = configure_logging()
MLFLOW_URI, MODEL_NAME, JENKINS_BASE, JENKINS_JOB, JENKINS_USER, JENKINS_TOKEN = load_configurations()
JENKINS_URL = f"{JENKINS_BASE}/job/{JENKINS_JOB}/buildWithParameters"
client = initialize_mlflow_client(MLFLOW_URI)


def get_jenkins_crumb_and_cookies():
    """
    Retrieves Jenkins crumb and session cookies for authentication.

    Returns:
        tuple: A tuple containing the crumb field name, crumb value, and requests Session object.

    Raises:
        requests.RequestException: If the crumb request fails.
    """
    crumb_url = f"{JENKINS_BASE}/crumbIssuer/api/json"
    session = requests.Session()
    resp = session.get(crumb_url, auth=(JENKINS_USER, JENKINS_TOKEN), timeout=10)
    resp.raise_for_status()
    crumb_json = resp.json()
    return crumb_json["crumbRequestField"], crumb_json["crumb"], session


def trigger_jenkins(model_name: str, version: str) -> bool:
    """
    Triggers a Jenkins pipeline with the specified model name and version.

    Args:
        model_name (str): Name of the model to deploy.
        version (str): Version of the model to deploy.

    Returns:
        bool: True if the Jenkins pipeline was triggered successfully, False otherwise.

    Raises:
        requests.RequestException: If the Jenkins API call fails.
    """
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
    """
    Continuously checks for newly promoted champion models in MLflow and triggers Jenkins deployment.

    Polls MLflow every 10 seconds to identify models in the production stage with the champion tag.
    If a new champion model is found and not yet deployed, triggers Jenkins and tags it as deployed.
    """
    while True:
        try:
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            prod_champion_versions = [
                v for v in versions
                if v.tags.get("stage", "").lower() == "production" and v.tags.get("champion", "").lower() == "true"
            ]
            if prod_champion_versions:
                prod_champion_versions.sort(key=lambda v: int(v.version), reverse=True)
                latest_version = prod_champion_versions[0]
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
        time.sleep(10)


def main():
    """
    Entry point for the model promotion watcher.

    Starts the continuous monitoring of MLflow for promoted champion models and triggers Jenkins deployments.
    """
    logger.info("Starting model promotion watcher...")
    check_model_promotion()


if __name__ == "__main__":
    main()