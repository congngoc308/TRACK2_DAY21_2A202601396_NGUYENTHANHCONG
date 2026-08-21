import json
import os
import unittest.mock as mock
import numpy as np
import pandas as pd
from src.train import train

FEATURE_NAMES = [
    "age",
    "workclass",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]


def _make_temp_data(tmp_path):
  """Tạo dataset nhỏ với cùng schema Adult để sử dụng trong test."""
  rng = np.random.default_rng(0)
  n = 200
  X = rng.random((n, len(FEATURE_NAMES)))
  y = rng.integers(0, 2, size=n)

  df = pd.DataFrame(X, columns=FEATURE_NAMES)
  df["target"] = y

  train_path = str(tmp_path / "train.csv")
  eval_path = str(tmp_path / "holdout.csv")

  df.iloc[:160].to_csv(train_path, index=False)
  df.iloc[160:].to_csv(eval_path, index=False)

  return train_path, eval_path


@mock.patch("mlflow.sklearn.log_model")
@mock.patch("mlflow.log_metric")
@mock.patch("mlflow.log_params")
@mock.patch("mlflow.start_run")
def test_train_returns_float(
    mock_run, mock_params, mock_metric, mock_model, tmp_path
):
  """Kiểm tra hàm train() trả về một số thực trong [0.0, 1.0]."""
  train_path, eval_path = _make_temp_data(tmp_path)
  f1 = train(
      {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
      data_path=train_path,
      eval_path=eval_path,
  )
  assert isinstance(f1, float)
  assert 0.0 <= f1 <= 1.0


@mock.patch("mlflow.sklearn.log_model")
@mock.patch("mlflow.log_metric")
@mock.patch("mlflow.log_params")
@mock.patch("mlflow.start_run")
def test_report_file_created(
    mock_run, mock_params, mock_metric, mock_model, tmp_path
):
  """Kiểm tra file outputs/report.json được tạo sau khi huấn luyện."""
  train_path, eval_path = _make_temp_data(tmp_path)
  train(
      {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
      data_path=train_path,
      eval_path=eval_path,
  )
  assert os.path.exists("outputs/report.json")
  with open("outputs/report.json") as f:
    data = json.load(f)
  assert "f1_score" in data and "accuracy" in data


@mock.patch("mlflow.sklearn.log_model")
@mock.patch("mlflow.log_metric")
@mock.patch("mlflow.log_params")
@mock.patch("mlflow.start_run")
def test_model_file_created(
    mock_run, mock_params, mock_metric, mock_model, tmp_path
):
  """Kiểm tra file models/model.joblib được tạo sau khi huấn luyện."""
  train_path, eval_path = _make_temp_data(tmp_path)
  train(
      {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
      data_path=train_path,
      eval_path=eval_path,
  )
  assert os.path.exists("models/model.joblib")