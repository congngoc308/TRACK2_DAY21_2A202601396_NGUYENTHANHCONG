import os
import boto3
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Income Prediction API")

# Lấy tên Bucket từ biến môi trường hoặc dùng mặc định
ARTIFACT_BUCKET = os.getenv("ARTIFACT_BUCKET", "income-lab-bucket-cong2026")
MODEL_KEY = "artifacts/current/model.joblib"
MODEL_PATH = os.path.expanduser("/tmp/model.joblib")

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

model = None


def download_model():
  """Tải file model.joblib từ S3 về máy khi server khởi động."""
  global model
  try:
    # TODO 1 & 2 & 3: Tải model từ AWS S3
    s3 = boto3.client("s3")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    s3.download_file(ARTIFACT_BUCKET, MODEL_KEY, MODEL_PATH)

    # TODO 4: In thông báo thành công
    print("Model đã được tải xuống từ S3 thành công.")
    model = joblib.load(MODEL_PATH)
  except Exception as e:
    print(f"Lỗi tải model từ S3: {e}")
    # Fallback nếu file model cục bộ đã có sẵn
    if os.path.exists("models/model.joblib"):
      model = joblib.load("models/model.joblib")


@app.on_event("startup")
def startup_event():
  download_model()


class ScoreRequest(BaseModel):
  features: list[float]


@app.get("/healthz")
def healthz():
  """TODO 5: Endpoint kiểm tra sức khỏe server (trả về status ok)."""
  return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest):
  """Endpoint suy luận chính phục vụ dự đoán."""
  global model
  if model is None:
    download_model()
    if model is None:
      raise HTTPException(
          status_code=503, detail="Mô hình chưa sẵn sàng (chưa tải được model)"
      )

  # TODO 6: Kiểm tra số lượng đặc trưng đầu vào (phải đúng 10 đặc trưng)
  if len(req.features) != 10:
    raise HTTPException(
        status_code=400,
        detail=f"Cần chính xác 10 đặc trưng, nhận được {len(req.features)}",
    )

  # TODO 7: Gọi model.predict để dự đoán
  df = pd.DataFrame([req.features], columns=FEATURE_NAMES)
  prediction = int(model.predict(df)[0])

  # TODO 8: Trả về dict chứa prediction và label tương ứng
  label = "thu_nhap_cao" if prediction == 1 else "thu_nhap_thap"
  return {"prediction": prediction, "label": label}


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=8080)