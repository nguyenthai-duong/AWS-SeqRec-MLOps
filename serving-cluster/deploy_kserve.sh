#!/bin/bash
set -e

### ===  Hàm chờ Pod trong namespace lên Ready ===
wait_for_pods_ready() {
    NS=$1
    echo -e "\n⏳ Chờ Pod trong namespace '$NS' Ready..."
    until kubectl get pods -n $NS &> /dev/null; do sleep 2; done
    while true; do
        NOT_READY=$(kubectl -n $NS get pods --no-headers 2>/dev/null \
            | awk '{split($2,a,"/"); if(a[1]!=a[2]) print}' | wc -l)
        ALL=$(kubectl -n $NS get pods --no-headers 2>/dev/null | wc -l)
        if [[ $ALL -gt 0 && $NOT_READY -eq 0 ]]; then
            echo "✅ $ALL Pod trong '$NS' đã Ready"; break
        else
            echo "  $((ALL-NOT_READY))/$ALL Ready. Đợi thêm..."; sleep 5
        fi
    done
}

### 1. Cleanup cũ
echo -e "\n🧹 Xoá namespace cũ nếu có..."
kubectl delete ns cert-manager knative-serving istio-system kserve --ignore-not-found

### 2. Tạo namespace
echo -e "\n🔹 Tạo namespace..."
for ns in cert-manager knative-serving istio-system kserve; do
  kubectl create ns $ns || true
done

### 3. Cert-Manager
echo -e "\n🔹 Cài Cert-Manager v1.13.3..."
kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.13.3/cert-manager.yaml
wait_for_pods_ready cert-manager

### 4. Knative Serving
echo -e "\n🔹 Cài Knative Serving CRDs (v1.11.3)..."
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.11.3/serving-crds.yaml

echo -e "\n🔹 Cài Knative Serving Core (v1.11.3)..."
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.11.3/serving-core.yaml
wait_for_pods_ready knative-serving

### 5. Istio (via istioctl)
echo -e "\n🔹 Cài Istio (profile=demo) bằng istioctl..."
if ! command -v istioctl &> /dev/null; then
  curl -L https://istio.io/downloadIstio | ISTIO_VERSION=1.17.2 sh -
  export PATH="$PWD/istio-1.17.2/bin:$PATH"
fi
istioctl install --set profile=demo --skip-confirmation
wait_for_pods_ready istio-system

### 6. Knative net-Istio
echo -e "\n🔹 Cài net-Istio (knative-v1.11.3)..."
kubectl apply -f https://github.com/knative/net-istio/releases/download/knative-v1.11.3/net-istio.yaml
wait_for_pods_ready istio-system

echo -e "\n🔹 Cài net-Istio release (knative-v1.11.3)..."
kubectl apply -f https://github.com/knative/net-istio/releases/download/knative-v1.11.3/release.yaml
wait_for_pods_ready istio-system

### 7. KServe
echo -e "\n🔹 Cài KServe v0.12.0..."
kubectl apply -n kserve -f https://github.com/kserve/kserve/releases/download/v0.12.0/kserve.yaml
wait_for_pods_ready kserve

echo -e "\n🎉 Hoàn tất! All namespaces are ready."
