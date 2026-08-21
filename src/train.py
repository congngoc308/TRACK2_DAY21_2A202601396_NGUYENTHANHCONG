import os
import json
import joblib
import yaml
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn

F1_THRESHOLD = 0.65


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huấn luyện mô hình và ghi nhận kết quả vào MLflow.
    """
    # 1.6.1: Đọc dữ liệu huấn luyện và đánh giá
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # 1.6.2: Tách đặc trưng và nhãn target
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # 1.6.3: Bắt đầu MLflow run
    with mlflow.start_run():
        # 1.6.4: Log siêu tham số
        mlflow.log_params(params)

        # 1.6.5: Khởi tạo và huấn luyện GradientBoostingClassifier
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # 1.6.6: Đánh giá mô hình (F1 lớp dương, không truyền average)
        preds = model.predict(X_eval)
        f1 = float(f1_score(y_eval, preds))
        acc = float(accuracy_score(y_eval, preds))

        # 1.6.7: Log metrics lên MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)

        # 1.6.8: Log mô hình vào MLflow artifact
        mlflow.sklearn.log_model(model, "model")

        # 1.6.9: In kết quả ra terminal
        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")

        # 1.6.10: Lưu metrics ra outputs/report.json
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.json", "w") as f:
            json.dump({"f1_score": f1, "accuracy": acc}, f, indent=2)

        # 1.6.11: Lưu mô hình ra models/model.joblib
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

        # 1.6.12: Trả về F1-score
        return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)