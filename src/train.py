import json
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
import yaml

F1_THRESHOLD = 0.65


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
  """Huấn luyện mô hình và ghi nhận kết quả vào MLflow."""
  df_train = pd.read_csv(data_path)
  df_eval = pd.read_csv(eval_path)

  X_train = df_train.drop(columns=["target"])
  y_train = df_train["target"]
  X_eval = df_eval.drop(columns=["target"])
  y_eval = df_eval["target"]

  # Đặt tracking URI tương đối để tránh dính đường dẫn tuyệt đối Windows
  mlflow.set_tracking_uri("sqlite:///mlflow.db")

  with mlflow.start_run():
    mlflow.log_params(params)

    model = GradientBoostingClassifier(**params, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_eval)
    f1 = float(f1_score(y_eval, preds))
    acc = float(accuracy_score(y_eval, preds))

    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/report.json", "w") as f:
      json.dump({"f1_score": f1, "accuracy": acc}, f, indent=2)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.joblib")

    return f1


if __name__ == "__main__":
  with open("params.yaml") as f:
    params = yaml.safe_load(f)
  train(params)