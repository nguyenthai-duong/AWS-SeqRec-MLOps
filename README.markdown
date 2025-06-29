DATN-RecSys
A production-ready Recommender System pipeline.

🚀 Quickstart
Clone the repository and set up the environment:
# Clone the repository
git clone https://github.com/nguyenthai-duong/AWS-SeqRec-MLOps
cd AWS-SeqRec-MLOps

# Create and activate a Conda environment
conda create -n recsys_ops python=3.11 -y
conda activate recsys_ops

# Install uv and dependencies
pip install uv==0.6.2
uv sync --all-groups

# Install Jupyter kernel
python -m ipykernel install --user --name=datn-recsys --display-name="Python (datn-recsys)"

4. Install Pre-commit (Recommended for Auto-Linting)
make precommit

6. Check Code Style and Lint
Run style checks on all code and notebooks:
make style

7. Run Unit Tests
Execute unit tests to ensure code quality:
make test

Setup GitHub PR Agent
Configure the PR Agent for automated pull request reviews. Follow the instructions at: PR Agent.

Navigate to Settings → Secrets and variables → Actions → New repository secret:

Name: OPENAI_API_KEY
Value: Paste your OpenAI API Key


Create a label:

Go to Issues → Labels
Click New label
Name: pr-agent/review
Color: Use default or choose a custom color
Description: "Trigger PR Agent review on this PR"


Click Create label


Attach the pr-agent/review label when creating a Pull Request.



Setting Up Kind Cluster and Kubeflow Infrastructure
Install NVIDIA Docker Runtime
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo apt-get install -y nvidia-container-toolkit nvidia-docker2
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
EOF
sudo systemctl daemon-reexec
sudo systemctl restart docker

Verify NVIDIA Runtime
docker info | grep -i runtime

Expected Output:
 Runtimes: runc io.containerd.runc.v2 nvidia
 Default Runtime: nvidia

Create Kind Cluster with GPU Support
sudo sed -i '/accept-nvidia-visible-devices-as-volume-mounts/c\accept-nvidia-visible-devices-as-volume-mounts = true' /etc/nvidia-container-runtime/config.toml

kind create cluster --name datn-training1 --config - <<EOF
apiVersion: kind.x-k8s.io/v1alpha4
kind: Cluster
nodes:
- role: control-plane
  image: kindest/node:v1.24.0
  extraMounts:
    - hostPath: /dev/null
      containerPath: /var/run/nvidia-container-devices/all
EOF

helm repo add nvidia https://helm.ngc.nvidia.com/nvidia || true
helm repo update
helm install --wait --generate-name \
     -n gpu-operator --create-namespace \
     nvidia/gpu-operator --set driver.enabled=false

Verify GPU Operator
kubectl get pods -n gpu-operator

Install Kubeflow
RELEASE=v1.7.0-rc.0
git clone -b $RELEASE --depth 1 --single-branch https://github.com/kubeflow/manifests.git
cd manifests
while ! kustomize build example | kubectl apply -f -; do echo "Retrying to apply resources"; sleep 10; done

Fix Authservice Bug
The authservice-0 pod may fail to pull from gcr.io/arrikto/kubeflow/oidc-authservice:28c59ef. Rebuild the image:
kubectl delete pod authservice-0 -n istio-system --grace-period=0 --force
git clone https://github.com/arrikto/oidc-authservice.git
make docker-build
docker tag gcr.io/arrikto-playground/kubeflow/oidc-authservice:0c4ea9a nthaiduong83/oidc-authservice:0c4ea9a


Update manifests/common/oidc-authservice/base/statefulset.yaml to use nthaiduong83/oidc-authservice:0c4ea9a.
Change autoscaling/v2beta2 to autoscaling/v2 in the manifests.

Handle "Too Many Open Files" Error
Check current inotify limits:
sysctl fs.inotify.max_user_watches
sysctl fs.inotify.max_user_instances

Apply new limits:
echo "fs.inotify.max_user_watches=524288" | sudo tee /etc/sysctl.d/99-kubeflow.conf
echo "fs.inotify.max_user_instances=512" | sudo tee -a /etc/sysctl.d/99-kubeflow.conf
sudo sysctl -p /etc/sysctl.d/99-kubeflow.conf

Ray Cluster Setup
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
cd ray-cluster
docker build -t nthaiduong83/ray-cluster:v1 -f ray.Dockerfile .
kind load docker-image nthaiduong83/ray-cluster:v1 --name datn-training1
# Install CRDs and KubeRay operator v1.3.0
kubens kubeflow-user-example-com
helm install kuberay-operator kuberay/kuberay-operator --version 1.3.0
kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/master/deploy/local-path-storage.yaml
kubectl apply -f ray-pvc.yaml
kubectl apply -f ray-worker-pvc.yaml
helm install raycluster .
kubectl port-forward svc/raycluster-kuberay-head-svc 10001:10001 -n kubeflow-user-example-com

Install MLflow
docker build -t nthaiduong83/mlflow-kubeflow:v1 -f ./mlflow-stack/mlflow.Dockerfile .
kind load docker-image nthaiduong83/mlflow-kubeflow:v1 --name datn-training1
helm upgrade mlflow-stack ./mlflow-stack -n mlflow --install --create-namespace

Install Jenkins
docker build -f ./jenkins-stack/Dockerfile.jenkins -t nthaiduong83/jenkins-datn:v1 .
kind load docker-image nthaiduong83/jenkins-datn:v1 --name datn-training1
helm upgrade jenkins-stack ./jenkins-stack -n devops-tools --install --create-namespace

Retrieve Jenkins Admin Password
kubectl exec -n devops-tools -it jenkins-55f88b6b77-6xxnc -- cat /var/jenkins_home/secrets/initialAdminPassword

Example Output: 7f225be7156d4207a59ed6f4960140f2
Install the suggested plugins during Jenkins setup.
Integrate MLflow and Jenkins into Kubeflow Dashboard
kubectl get configmap centraldashboard-config -n kubeflow -o yaml > dashboard-config.yaml

Add the following to dashboard-config.yaml:
[
  {
    "type": "item",
    "link": "/mlflow/",
    "text": "MLflow",
    "icon": "icons:cached"
  },
  {
    "type": "item",
    "link": "/jenkins/",
    "text": "Jenkins",
    "icon": "icons:extension"
  }
]

Apply and restart:
kubectl apply -f dashboard-config.yaml
kubectl rollout restart deployment centraldashboard -n kubeflow

Access Kubeflow UI
kubectl port-forward svc/istio-ingressgateway 8000:80 -n istio-system

Data Pipeline Setup
Refer to ./data_pipeline_aws/README.md for detailed instructions.
Setup Serving Cluster (Local)
kind create cluster --name datn-serving --config - <<EOF
apiVersion: kind.x-k8s.io/v1alpha4
kind: Cluster
nodes:
- role: control-plane
  image: kindest/node:v1.26.3
  extraMounts:
    - hostPath: /dev/null
      containerPath: /var/run/nvidia-container-devices/all
EOF

helm repo add nvidia https://helm.ngc.nvidia.com/nvidia || true
helm repo update
helm install --wait --generate-name \
     -n gpu-operator --create-namespace \
     nvidia/gpu-operator --set driver.enabled=false

Feature Store Setup
MATERIALIZE_CHECKPOINT_TIME=$(uv run ./src/check_oltp_max_timestamp.py 2>&1 | awk -F'<ts>|</ts>' '{print $2}')
echo "MATERIALIZE_CHECKPOINT_TIME=$MATERIALIZE_CHECKPOINT_TIME"

Example Output: 2022-06-14 05:27:26.678000+00:00
cd feature_store
# Load AWS and Postgres credentials
export $(grep -v '^#' ../.env | xargs)
uv run feast apply
uv run feast materialize 2010-01-01T00:00:00 "$MATERIALIZE_CHECKPOINT_TIME"

docker build -t nthaiduong83/feature-store-api:v3 -f feature_store_api.Dockerfile .
docker push nthaiduong83/feature-store-api:v3

kubectl create ns api-feature-store
kubectl create secret generic aws-credentials --from-env-file=../.env --namespace=api-feature-store

kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl port-forward svc/feature-store-api-service 8005:80 -n api-feature-store

Feature Engineering Pipeline in Kubeflow
cd src/feature_engineer
kubens kubeflow-user-example-com
kubectl apply -f pvc.yaml
kubectl get pvc
kubectl apply -f copy-job.yaml
# Upload raw data to PVC
chmod +x copy-to-pvc.sh
./copy-to-pvc.sh
# Verify PVC contents
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: pvc-checker
spec:
  containers:
  - name: shell
    image: busybox
    command: ["sh", "-c", "sleep 3600"]
    volumeMounts:
    - name: data-volume
      mountPath: /data
  volumes:
  - name: data-volume
    persistentVolumeClaim:
      claimName: data-pvc
  restartPolicy: Never
EOF

kubectl exec -it pvc-checker -- sh
ls -lh /data
kubectl delete pod pvc-checker --force

docker build -f src/feature_engineer/feature_pipeline.Dockerfile -t nthaiduong83/feature_pipeline:v2 .
kind load docker-image nthaiduong83/feature_pipeline:v6 --name datn-training1

kubectl create secret generic aws-credentials --from-env-file=.env -n kubeflow-user-example-com

cd src/kfp_pipeline
uv run run_pipeline.py

Offline Caching with Redis
Deploy Qdrant and expose its port:
helm install qdrant ./qdrant --namespace kubeflow-user-example-com
kubectl port-forward svc/qdrant 6333:6333 -n kubeflow-user-example-com

In the training cluster, expose MLflow and MinIO ports for offline caching:
kubectl port-forward --address 127.0.0.1 svc/minio-service 9010:9000 -n mlflow
kubectl port-forward svc/mlflow-tracking-service -n mlflow 5002:5000

Switch to the serving cluster and deploy Redis:
kubectx kind-datn-serving
kubectl create ns cache
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update
helm install redis bitnami/redis \
  --version 21.0.2 \
  --namespace cache \
  --set-string auth.password=123456 \
  --set master.service.type=LoadBalancer

kubectl port-forward svc/redis-master 6379:6379 -n cache

Run the offline caching pipeline in the notebook: src/caching_offline/load2redis.ipynb

## Building serving cluster
### Build serving cluster using EKS
### [Option]Build serving cluster using kind in local 
kind create cluster --name datn-serving --config - <<EOF
apiVersion: kind.x-k8s.io/v1alpha4
kind: Cluster
nodes:
- role: control-plane
  image: kindest/node:v1.26.3
  extraMounts:
    - hostPath: /dev/null
      containerPath: /var/run/nvidia-container-devices/all
EOF
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia || true
helm repo update
helm install --wait --generate-name \
     -n gpu-operator --create-namespace \
     nvidia/gpu-operator --set driver.enabled=false
#### Check:
docker info | grep -i runtime
 Runtimes: io.containerd.runc.v2 nvidia runc
 Default Runtime: nvidia





kubectx kind-datn-serving
cd serving-cluster
./deploy_kserve.sh

docker build -t tritonserver-datn:v4 . -f Dockerfile.triton

docker run --gpus=1 --rm \
  -v /home/duong/Documents/datn1/src/model_ranking_sequence/model_repository:/models \
  -p8008:8000 -p8001:8001 -p8002:8002 \
  tritonserver-datn:v4 --model-repository=/models --log-verbose=2

kind load docker-image tritonserver-datn:v4 --name datn-serving

### Jenkins pipeline
kind get kubeconfig --name datn-serving --internal > kubeconfig-serving.yaml
Cài plugin
docker pipeline, kubernetes cli, stage view

Tạo pipeline, from scm
điền link git
tạo credential
chọn Username with password

Chạy pipeline