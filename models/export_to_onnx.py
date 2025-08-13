from ultralytics import YOLO
import ultralytics
import os
# Load a model
# model = YOLO("yolo11n.pt")  # load an official model
# 현재 파일 기준으로 상대경로 사용 (예: weights 폴더가 같은 디렉토리에 있을 때)
base_dir = os.path.dirname(__file__)  # 현재 파일 위치
weights_path = os.path.join(base_dir, "weights", "best.pt")

# 모델 로드
model = YOLO(weights_path)
# model = YOLO("weights/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx")