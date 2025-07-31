pipeline {
  agent any

  environment {
    CONDA_ENV = "datn"
    AWS_DEFAULT_REGION = "ap-southeast-1"
    S3_MODEL_REPO = "s3://recsys-triton-repo/"
    MLFLOW_S3_ENDPOINT_URL = "http://minio-service.mlflow.svc.cluster.local:9000"
    MLFLOW_TRACKING_URI = "http://mlflow-tracking-service.mlflow.svc.cluster.local:5000"
    PATH = "$HOME/.local/bin:$PATH"
    MODEL_NAME = "${params.MODEL_NAME ?: 'seq_tune_v1_sequence_rating'}"
    MODEL_VERSION = "${params.MODEL_VERSION ?: 'latest'}"
  }

  stages {

    stage('Convert to Triton repo') {
      steps {
        sh '''
        bash -c "
          source /opt/conda/etc/profile.d/conda.sh
          conda activate ${CONDA_ENV}
          pip install tqdm
          python src/model_ranking_sequence/convert2onnx_and_build_triton.py
        "
        '''
      }
    }

    stage('Test Triton repo and Install awscli') {
      steps {
        sh '''
        bash -c "
          source /opt/conda/etc/profile.d/conda.sh
          conda activate ${CONDA_ENV}
          # Check if model repository exists and has required structure
          if [ ! -d "./src/model_ranking_sequence/model_repository" ]; then
            echo "Error: Model repository directory not found"
            exit 1
          fi
          # Check for ONNX model file
          if [ ! -f "./src/model_ranking_sequence/model_repository/${MODEL_NAME}/1/model.onnx" ]; then
            echo "Error: ONNX model file not found"
            exit 1
          fi
          # Check for config.pbtxt
          if [ ! -f "./src/model_ranking_sequence/model_repository/${MODEL_NAME}/config.pbtxt" ]; then
            echo "Error: Triton config file not found"
            exit 1
          fi
          echo "Triton repository structure validated successfully"

          pip install awscli
        "
        '''
      }
    }

    stage('Upload model repo to S3') {
      steps {
        withCredentials([aws(accessKeyVariable: 'AWS_ACCESS_KEY_ID', secretKeyVariable: 'AWS_SECRET_ACCESS_KEY', credentialsId: 'aws-credentials')]) {
          sh '''
          aws s3 rm ${S3_MODEL_REPO} --recursive || true
          aws s3 sync ./src/model_ranking_sequence/model_repository/ ${S3_MODEL_REPO}
          touch .keep
          aws s3 cp .keep ${S3_MODEL_REPO}ensemble/1/.keep
          '''
        }
      }
    }

    stage('Deploy to KServe') {
      steps {
        withCredentials([aws(accessKeyVariable: 'AWS_ACCESS_KEY_ID', secretKeyVariable: 'AWS_SECRET_ACCESS_KEY', credentialsId: 'aws-credentials')]) {
          sh '''
          export KUBECONFIG="./serving-cluster/kubeconfig-serving.yaml"
          kubectl apply -f ./serving-cluster/inferenceservice-triton-gpu.yaml --validate=false
          sleep 5
          kubectl delete pod -n kserve -l serving.kserve.io/inferenceservice=recsys-triton || true
          '''
        }
      }
    }
  }
}