# file2_fixed.py
import os
import mlflow
import mlflow.sklearn
from mlflow.exceptions import RestException

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==== Config ====
EXPERIMENT_NAME = "Mlflow_exp23"
LOCAL_MODEL_DIR = "tmp_models/rf_model"        # local save directory
ARTIFACT_MODEL_PATH = "model"                  # artifact path in run
CONF_MATRIX_FILE = "confusion_matrix.png"

os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# ==== Data & train ====
wine = load_wine()
X, y = wine.data, wine.target
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_estimators = 6
max_depth = 4

rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
rf.fit(X_train, y_train)
y_pred = rf.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"accuracy: {acc:.6f}")

# ==== save confusion matrix image ====
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, xticklabels=wine.target_names, yticklabels=wine.target_names)
plt.title("confusion matrix")
plt.xlabel("predicted")
plt.ylabel("actual")
plt.savefig(CONF_MATRIX_FILE)
plt.close()

# ==== MLflow logging (no registry calls) ====
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run() as run:
    run_id = run.info.run_id
    print("Started run:", run_id)

    # log metrics & params
    mlflow.log_metric("accuracy", float(acc))
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_artifact(CONF_MATRIX_FILE)  # log the confusion matrix image

    # *** DO NOT CALL mlflow.sklearn.log_model(...) HERE ***
    # Save the model locally and upload the directory as artifacts (works with providers that don't support registry)
    try:
        mlflow.sklearn.save_model(rf, LOCAL_MODEL_DIR)
        mlflow.log_artifacts(LOCAL_MODEL_DIR, artifact_path=ARTIFACT_MODEL_PATH)
        print(f"Model saved locally to '{LOCAL_MODEL_DIR}' and uploaded to artifacts under '{ARTIFACT_MODEL_PATH}'.")
    except Exception as e:
        print("Failed to save/log model artifacts:", type(e).__name__, e)

    # set tags (optional)
    mlflow.set_tags({"Author": "Ganesh", "Project": "placement prediction"})

print("Run finished. Check run artifacts in your DagsHub run UI.")
