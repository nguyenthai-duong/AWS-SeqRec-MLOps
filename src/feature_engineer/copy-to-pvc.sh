#!/bin/bash

NAMESPACE="kubeflow-user-example-com"
JOB_YAML="copy-job.yaml"

if [ ! -f "$JOB_YAML" ]; then
  echo "Error: $JOB_YAML not found"
  exit 1
fi

echo "Checking PVC data-pvc..."
if ! kubectl get pvc data-pvc -n $NAMESPACE &>/dev/null; then
  echo "Error: PVC data-pvc not found in namespace $NAMESPACE"
  exit 1
fi
echo "PVC data-pvc found"

echo "Checking local files..."
for file in /home/duong/Documents/datn1/data/raw_meta.parquet; do
  if [ ! -f "$file" ]; then
    echo "Error: File $file not found"
    exit 1
  fi
done
echo "All local files found"

echo "Creating Job copy-job..."
kubectl apply -f $JOB_YAML
if [ $? -ne 0 ]; then
  echo "Error: Failed to apply $JOB_YAML"
  exit 1
fi

echo "Waiting for pod to be ready..."
for i in {1..12}; do # Chờ tối đa 60 giây (12 * 5s)
  POD_NAME=$(kubectl get pods -n $NAMESPACE -l job-name=copy-job -o jsonpath="{.items[0].metadata.name}" 2>/dev/null)
  if [ -n "$POD_NAME" ]; then
    POD_STATUS=$(kubectl get pod $POD_NAME -n $NAMESPACE -o jsonpath="{.status.phase}" 2>/dev/null)
    if [ "$POD_STATUS" == "Running" ]; then
      echo "Pod $POD_NAME is Running"
      break
    fi
  fi
  echo "Pod not ready yet, waiting 5 seconds..."
  sleep 5
done

if [ -z "$POD_NAME" ] || [ "$POD_STATUS" != "Running" ]; then
  echo "Error: Pod not created or not Running after 60 seconds"
  kubectl describe job copy-job -n $NAMESPACE
  kubectl get pods -n $NAMESPACE -l job-name=copy-job
  exit 1
fi

echo "Copying files to PVC..."
kubectl cp /home/duong/Documents/datn/data/raw_meta.parquet $POD_NAME:/mnt/data/raw_meta.parquet -n $NAMESPACE

echo "Verifying files in PVC..."
kubectl exec -it $POD_NAME -n $NAMESPACE -- ls /mnt/data
if [ $? -ne 0 ]; then
  echo "Error: Failed to verify files in PVC"
  kubectl logs $POD_NAME -n $NAMESPACE
  exit 1
fi

echo "Cleaning up..."
kubectl delete job copy-job -n $NAMESPACE