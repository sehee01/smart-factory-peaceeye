from ultralytics import YOLO

# 모델 로드
model = YOLO("weights/best_yolo12m.pt")

# TensorRT로 직접 변환 (메타데이터 자동 포함)
model.export(format="engine", 
            half=True,  # FP16
            workspace=4,  # 4GB workspace
            verbose=True)