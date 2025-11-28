"""
Machine Learning Training Pipeline
===================================

This example demonstrates a complete ML training workflow including
data preparation, model training, evaluation, and deployment.

Prerequisites:
--------------
None (uses synthetic data and simple models)

Concepts Covered:
-----------------
1. ML pipeline workflow
2. Data preprocessing and feature engineering
3. Model training and evaluation
4. Hyperparameter tuning
5. Model versioning and deployment

Expected Output:
----------------
=== ML Training Pipeline ===

📊 Step 1: Data Preparation
   Loading training data...
   ✅ Loaded 1000 samples
   ✅ Features: 10, Labels: binary classification

🔧 Step 2: Feature Engineering
   Creating features...
   ✅ Normalized features
   ✅ Created polynomial features
   ✅ Train/validation split: 800/200

🤖 Step 3: Model Training
   Training model...
   Epoch 1/3: loss=0.523
   Epoch 2/3: loss=0.412
   Epoch 3/3: loss=0.358
   ✅ Training complete

📈 Step 4: Model Evaluation
   Evaluating on validation set...
   ✅ Accuracy: 0.89
   ✅ Precision: 0.87
   ✅ Recall: 0.91

🎯 Step 5: Hyperparameter Tuning
   Testing configuration 1/3...
   Testing configuration 2/3...
   Testing configuration 3/3...
   ✅ Best config: {'learning_rate': 0.01, 'batch_size': 64}
   ✅ Best accuracy: 0.92

💾 Step 6: Model Deployment
   Saving model...
   ✅ Model saved to: model_v1.pkl
   ✅ Deployment ready

=== Summary ===
Pipeline Status: SUCCESS
Training Samples: 1000
Validation Accuracy: 0.92
Model Version: v1
✅ ML pipeline completed successfully
"""

import random
import time

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.workflow import workflow


def main():
    """Run ML training pipeline."""
    print("=== ML Training Pipeline ===\n")

    with workflow("ml_training") as ctx:

        @task(inject_context=True)
        def prepare_data(context: TaskExecutionContext):
            """Step 1: Load and prepare training data."""
            print("📊 Step 1: Data Preparation")
            print("   Loading training data...")
            time.sleep(0.2)

            # Simulate loading data
            num_samples = 1000
            num_features = 10

            print(f"   ✅ Loaded {num_samples} samples")
            print(f"   ✅ Features: {num_features}, Labels: binary classification\n")

            # Store in channel
            channel = context.get_channel()
            channel.set("num_samples", num_samples)
            channel.set("num_features", num_features)
            channel.set("data_loaded", True)

        @task(inject_context=True)
        def engineer_features(context: TaskExecutionContext):
            """Step 2: Feature engineering and preprocessing."""
            print("🔧 Step 2: Feature Engineering")
            print("   Creating features...")
            time.sleep(0.2)

            channel = context.get_channel()
            num_samples = channel.get("num_samples")

            # Simulate feature engineering
            train_size = int(num_samples * 0.8)
            val_size = num_samples - train_size

            print("   ✅ Normalized features")
            print("   ✅ Created polynomial features")
            print(f"   ✅ Train/validation split: {train_size}/{val_size}\n")

            channel.set("train_size", train_size)
            channel.set("val_size", val_size)
            channel.set("features_ready", True)

        @task(inject_context=True)
        def train_model(context: TaskExecutionContext):
            """Step 3: Train the model."""
            print("🤖 Step 3: Model Training")
            print("   Training model...")

            # Simulate training epochs
            num_epochs = 3
            initial_loss = 0.6
            loss = 0.0

            for epoch in range(1, num_epochs + 1):
                time.sleep(0.15)
                # Simulate decreasing loss
                loss = initial_loss * (0.6 ** epoch) + random.uniform(0, 0.05)
                print(f"   Epoch {epoch}/{num_epochs}: loss={loss:.3f}")

            print("   ✅ Training complete\n")

            channel = context.get_channel()
            channel.set("model_trained", True)
            channel.set("final_loss", loss)

        @task(inject_context=True)
        def evaluate_model(context: TaskExecutionContext):
            """Step 4: Evaluate model performance."""
            print("📈 Step 4: Model Evaluation")
            print("   Evaluating on validation set...")
            time.sleep(0.2)

            # Simulate metrics
            accuracy = 0.89
            precision = 0.87
            recall = 0.91

            print(f"   ✅ Accuracy: {accuracy:.2f}")
            print(f"   ✅ Precision: {precision:.2f}")
            print(f"   ✅ Recall: {recall:.2f}\n")

            channel = context.get_channel()
            channel.set("accuracy", accuracy)
            channel.set("precision", precision)
            channel.set("recall", recall)

        @task(inject_context=True)
        def tune_hyperparameters(context: TaskExecutionContext):
            """Step 5: Hyperparameter tuning."""
            print("🎯 Step 5: Hyperparameter Tuning")

            # Test different configurations
            configs = [
                {"learning_rate": 0.001, "batch_size": 32},
                {"learning_rate": 0.01, "batch_size": 64},
                {"learning_rate": 0.1, "batch_size": 128},
            ]

            best_accuracy = 0
            best_config = None

            for i, config in enumerate(configs, 1):
                print(f"   Testing configuration {i}/{len(configs)}...")
                time.sleep(0.1)

                # Simulate evaluation
                accuracy = 0.85 + random.uniform(0, 0.1)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_config = config

            print(f"   ✅ Best config: {best_config}")
            print(f"   ✅ Best accuracy: {best_accuracy:.2f}\n")

            channel = context.get_channel()
            channel.set("best_config", best_config)
            channel.set("best_accuracy", best_accuracy)

        @task(inject_context=True)
        def deploy_model(context: TaskExecutionContext):
            """Step 6: Deploy the trained model."""
            print("💾 Step 6: Model Deployment")
            print("   Saving model...")
            time.sleep(0.2)

            model_version = "v1"
            model_path = f"model_{model_version}.pkl"

            print(f"   ✅ Model saved to: {model_path}")
            print("   ✅ Deployment ready\n")

            channel = context.get_channel()
            channel.set("model_version", model_version)
            channel.set("deployment_status", "SUCCESS")

        @task(inject_context=True)
        def generate_report(context: TaskExecutionContext):
            """Generate final pipeline report."""
            channel = context.get_channel()

            print("=== Summary ===")
            print(f"Pipeline Status: {channel.get('deployment_status')}")
            print(f"Training Samples: {channel.get('num_samples')}")
            print(f"Validation Accuracy: {channel.get('best_accuracy'):.2f}")
            print(f"Model Version: {channel.get('model_version')}")
            print("✅ ML pipeline completed successfully")

        # Define pipeline workflow
        prepare_data >> engineer_features >> train_model >> evaluate_model >> tune_hyperparameters >> deploy_model >> generate_report

        # Execute pipeline
        ctx.execute("prepare_data")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **ML Pipeline Structure**
#    Data → Features → Train → Evaluate → Tune → Deploy
#    - Clear sequential flow
#    - Each stage builds on previous results
#    - Reproducible and maintainable
#
# 2. **Data Sharing via Channels**
#    - Store data, metrics, and artifacts in channel
#    - Pass information between pipeline stages
#    - No need for file-based intermediate storage
#
# 3. **Modularity**
#    - Each step is an independent task
#    - Easy to modify individual components
#    - Can replace/upgrade stages independently
#
# 4. **Real-World ML Workflow**
#    ✅ Data loading and validation
#    ✅ Feature preprocessing
#    ✅ Model training with progress tracking
#    ✅ Evaluation metrics
#    ✅ Hyperparameter optimization
#    ✅ Model versioning and deployment
#
# 5. **Production Considerations**
#    - Add data validation
#    - Implement model checkpointing
#    - Track experiments (MLflow, W&B)
#    - Version datasets and models
#    - Monitor training progress
#
# ============================================================================
# Production Enhancements:
# ============================================================================
#
# **Data Validation**:
# @task
# def validate_data(context):
#     data = load_data()
#     assert data.shape[0] > 0, "Empty dataset"
#     assert not data.isnull().any().any(), "Missing values"
#     check_data_distribution(data)
#
# **Experiment Tracking**:
# @task
# def train_with_tracking(context):
#     import mlflow
#     with mlflow.start_run():
#         model = train_model()
#         mlflow.log_params(config)
#         mlflow.log_metrics(metrics)
#         mlflow.sklearn.log_model(model, "model")
#
# **Distributed Training**:
# - Use Redis backend for distributed execution
# - Train multiple models in parallel
# - Aggregate results for ensemble
#
# **Model Monitoring**:
# @task
# def monitor_model(context):
#     predictions = model.predict(new_data)
#     check_prediction_drift(predictions)
#     check_feature_drift(new_data)
#     alert_if_performance_degrades()
#
# **A/B Testing**:
# @task
# def ab_test_models(context):
#     model_a = load_model("v1")
#     model_b = load_model("v2")
#     compare_on_test_set(model_a, model_b)
#     select_winner()
#
# ============================================================================
# Integration with ML Frameworks:
# ============================================================================
#
# **PyTorch**:
# @task
# def train_pytorch_model(context):
#     model = create_model()
#     optimizer = torch.optim.Adam(model.parameters())
#     for epoch in range(num_epochs):
#         loss = train_epoch(model, dataloader, optimizer)
#         context.get_channel().set(f"loss_epoch_{epoch}", loss)
#
# **TensorFlow/Keras**:
# @task
# def train_keras_model(context):
#     model = tf.keras.Sequential([...])
#     model.compile(optimizer='adam', loss='binary_crossentropy')
#     history = model.fit(X_train, y_train, epochs=10)
#     context.get_channel().set("history", history.history)
#
# **Scikit-learn**:
# @task
# def train_sklearn_model(context):
#     from sklearn.ensemble import RandomForestClassifier
#     model = RandomForestClassifier()
#     model.fit(X_train, y_train)
#     score = model.score(X_val, y_val)
#     context.get_channel().set("model", model)
#
# ============================================================================
