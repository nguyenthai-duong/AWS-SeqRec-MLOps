pipeline {
  agent any

  environment {
    CONDA_ENV = "datn"
    AWS_ACCESS_KEY_ID="AKIA3TD2SE4D4EJQWMMS"
    AWS_SECRET_ACCESS_KEY="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"
    AWS_DEFAULT_REGION = "ap-southeast-1"
    S3_MODEL_REPO = "s3://datn-onnx/"
    MLFLOW_S3_ENDPOINT_URL = "http://minio-service.mlflow.svc.cluster.local:9000"
    MLFLOW_TRACKING_URI = "http://mlflow-tracking-service.mlflow.svc.cluster.local:5000"
    PATH = "$HOME/.local/bin:$PATH"
    MODEL_NAME = "${params.MODEL_NAME ?: 'seq_tune_v1_sequence_rating'}"
    MODEL_VERSION = "${params.MODEL_VERSION ?: 'latest'}"
  }

  stages {
    stage('Clone code') {
      steps {
        echo 'Code đã tự clone từ SCM'
      }
    }

    stage('Convert to Triton repo') {
      steps {
        sh '''
        bash -c "
          source /opt/conda/etc/profile.d/conda.sh
          conda activate ${CONDA_ENV}
          pip install tqdm
          python src/model_ranking_sequence/convert2onnx.py
        "
        '''
      }
    }

    stage('Install awscli') {
      steps {
        sh '''
        pip install awscli
        '''
      }
    }

    stage('Upload model repo to S3') {
      steps {
          sh '''
          aws s3 rm ${S3_MODEL_REPO} --recursive || true
          aws s3 sync ./src/model_ranking_sequence/model_repository/ ${S3_MODEL_REPO}
          touch .keep
          aws s3 cp .keep ${S3_MODEL_REPO}ensemble/1/.keep
          '''
      }
    }

    stage('Deploy to KServe') {
      steps {
        sh '''
          export KUBECONFIG="./serving-cluster/kubeconfig-serving.yaml"
          kubectl version
          kubectl create ns kserve || true
          kubectl get ns

          kubectl apply -f ./serving-cluster/inferenceservice-triton-gpu.yaml --validate=false
          sleep 5
          kubectl delete pod -n kubeflow-user-example-com -l serving.kserve.io/inferenceservice=recsys-triton || true
        '''
      }
    }

  }
}