# Kubernetes による自動スケーリング

**作成日:** 2025-01-24
**ステータス:** Draft
**コンセプト:** Simple Queue-Based Autoscaling

---

## 1. 設計概要

### 1.1 目的

Kubernetes標準のHPAを使い、Redisキュー深度に基づいてWorker数を自動調整する。

### 1.2 基本方針

**設計原則:**
- Kubernetes標準のHPA（Horizontal Pod Autoscaler）を使用
- サードパーティツール（KEDA等）は使わない
- シンプルな実装（Prometheus Adapter + HPA）

**スケーリングロジック:**

```
desired_workers = ceil(queue_depth / tasks_per_worker)
```

例: キューに100タスク、1 Workerあたり10タスク → 10 Workers

---

## 2. アーキテクチャ

```
Producer → Redis Queue ← Worker Pods
                ↓
         Queue Metrics → Prometheus → Prometheus Adapter
                                            ↓
                                          HPA → Scale Workers
```

**コンポーネント:**

1. **Redis**: タスクキュー（既存）
2. **Metrics Exporter**: キュー深度をPrometheusメトリクスとして公開
3. **Prometheus**: メトリクス収集（既存を想定）
4. **Prometheus Adapter**: PrometheusメトリクスをKubernetes Custom Metricsに変換
5. **HPA**: キュー深度で自動スケール
6. **Worker Deployment**: Kubernetes Deploymentで管理

---

## 3. 実装

### 3.1 Metrics Exporter（シンプル版）

```python
# graflow/metrics/queue_exporter.py
from prometheus_client import Gauge, generate_latest
from flask import Flask, Response
import redis, os

app = Flask(__name__)
gauge = Gauge('graflow_queue_depth', 'Queue depth', ['key_prefix'])
r = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379)

@app.route('/metrics')
def metrics():
    prefix = os.getenv('KEY_PREFIX', 'graflow')
    depth = r.llen(f"{prefix}:queue")
    gauge.labels(key_prefix=prefix).set(depth)
    return Response(generate_latest(), mimetype='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

### 3.2 Kubernetes マニフェスト

**Worker Deployment:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: graflow-worker
spec:
  replicas: 2  # 手動スケーリング: この値を変更
  selector:
    matchLabels:
      app: graflow-worker
  template:
    metadata:
      labels:
        app: graflow-worker
    spec:
      containers:
      - name: worker
        image: graflow-worker:latest
        args:
          - --worker-id=$(POD_NAME)
          - --redis-host=redis-service
          - --redis-key-prefix=graflow
        env:
          - name: POD_NAME
            valueFrom:
              fieldRef:
                fieldPath: metadata.name
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "2000m"
            memory: "2Gi"
```

**HPA設定:**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: graflow-worker-hpa
spec:
  scaleTargetRef:
    kind: Deployment
    name: graflow-worker
  minReplicas: 1
  maxReplicas: 20
  metrics:
  - type: External
    external:
      metric:
        name: graflow_queue_depth
      target:
        type: AverageValue
        averageValue: "10"  # 1 Workerあたり10タスク
```

**Note:** Prometheus Adapterの設定が必要（後述）

---

## 4. デプロイ手順

### 4.1 前提条件

- Kubernetes クラスタ（v1.23+）
- Prometheus（既存を想定、なければHelm等で導入）
- kubectl CLI

### 4.2 セットアップ

**1. Prometheus Adapter インストール（Helm）:**

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install prometheus-adapter prometheus-community/prometheus-adapter \
  --set prometheus.url=http://prometheus-server.monitoring.svc \
  --set prometheus.port=80
```

**2. Metrics Exporter デプロイ:**

```bash
kubectl apply -f metrics-exporter.yaml
```

**3. Worker Deployment デプロイ:**

```bash
kubectl apply -f worker-deployment.yaml
```

**4. HPA 作成:**

```bash
kubectl apply -f hpa.yaml
```

**5. 動作確認:**

```bash
# HPA状態確認
kubectl get hpa graflow-worker-hpa

# メトリクスが取得できているか確認
kubectl get --raw /apis/external.metrics.k8s.io/v1beta1/namespaces/default/graflow_queue_depth

# Worker Pods確認
kubectl get pods -l app=graflow-worker --watch
```

---

## 5. 運用

### 5.1 モニタリング

**HPAの状態確認:**
```bash
# HPA状態
kubectl get hpa graflow-worker-hpa

# 詳細表示（スケーリングイベント含む）
kubectl describe hpa graflow-worker-hpa

# リアルタイム監視
watch kubectl get hpa graflow-worker-hpa
```

**キュー深度確認:**
```bash
# Redis CLI経由
kubectl exec -it deployment/redis -- redis-cli LLEN "graflow:queue"

# Metrics Exporter経由
kubectl port-forward svc/metrics-exporter 8000:8000
curl http://localhost:8000/metrics | grep graflow_queue_depth
```

**Worker状態確認:**
```bash
kubectl get pods -l app=graflow-worker
kubectl top pods -l app=graflow-worker  # CPU/メモリ使用率
```

### 5.2 手動スケーリング（緊急時）

HPAが自動でスケールしますが、緊急時やテスト時には手動でも調整可能:

```bash
# 一時的にHPAを無効化
kubectl delete hpa graflow-worker-hpa

# 手動でスケール
kubectl scale deployment graflow-worker --replicas=10

# HPAを再度有効化
kubectl apply -f hpa.yaml
```

**注意:** HPAが有効な状態で手動スケールしても、すぐにHPAが上書きします。

### 5.3 トラブルシューティング

**HPAが動作しない:**
```bash
# HPA状態確認
kubectl describe hpa graflow-worker-hpa

# メトリクスが取得できているか確認
kubectl get --raw /apis/external.metrics.k8s.io/v1beta1/namespaces/default/graflow_queue_depth

# Prometheus Adapterログ確認
kubectl logs -n monitoring deployment/prometheus-adapter

# Metrics Exporterログ確認
kubectl logs deployment/metrics-exporter
```

**Workerが起動しない:**
```bash
kubectl logs deployment/graflow-worker
kubectl describe pod <pod-name>
```

**スケールが遅い:**

HPAのデフォルト設定では安全のためゆっくりスケールします。必要に応じてbehavior設定を調整:

```yaml
spec:
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 0      # すぐにスケールアップ
      policies:
      - type: Percent
        value: 100                        # 一度に2倍まで
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300    # 5分待ってからスケールダウン
```

---

## 6. 設計思想

**採用する技術:**
- ✅ Kubernetes標準のHPA（Horizontal Pod Autoscaler）
- ✅ Prometheus + Prometheus Adapter（標準的なメトリクスパイプライン）
- ✅ シンプルなMetrics Exporter（~15行）

**実装しないもの（YAGNI）:**
- ❌ KEDA（標準HPAで十分）
- ❌ 複雑なbehavior設定（必要になったら追加）
- ❌ マルチテナント専用設定（key_prefixで分離可能）
- ❌ 優先度ベーススケーリング（必要になったら追加）

**設計原則:**
1. **Kubernetes標準機能を優先**: サードパーティツールは避ける
2. **シンプル**: 必要最小限の構成
3. **保守性**: 標準的な構成なので引き継ぎやすい

---

## 7. まとめ

**本設計の特徴:**
- Kubernetes標準のHPAを使用した自動スケーリング
- Prometheus Adapterでキュー深度をKubernetesメトリクスに変換
- シンプルな構成で保守しやすい

**導入のメリット:**
- ワークロード変動に自動対応
- 手動スケーリング不要
- コスト最適化（アイドル時はminReplicas、ピーク時は自動増強）

**推奨セットアップ:**
```
本番環境:
  minReplicas: 2 (冗長性確保)
  maxReplicas: 20-50 (ワークロードに応じて)
  averageValue: 10 (1 Workerあたり10タスク)

開発環境:
  minReplicas: 1
  maxReplicas: 10
  averageValue: 20 (コスト削減)
```
