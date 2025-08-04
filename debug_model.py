from ultralytics import YOLO
import cv2
import numpy as np
import torch

def debug_model(model_path, test_image_path=None):
    """모델 디버깅 함수"""
    print(f"=== DEBUGGING MODEL: {model_path} ===")
    
    try:
        # 1. 모델 로드
        model = YOLO(model_path)
        print(f"✓ Model loaded successfully")
        print(f"Model class names: {model.names}")
        print(f"Number of classes: {len(model.names)}")
        
        # 2. 간단한 테스트 이미지 생성
        if test_image_path:
            test_image = cv2.imread(test_image_path)
            print(f"✓ Loaded test image: {test_image_path}")
        else:
            # 사람 모양의 간단한 테스트 이미지 생성
            test_image = np.zeros((640, 640, 3), dtype=np.uint8)
            # 사람 모양 그리기 (간단한 직사각형)
            cv2.rectangle(test_image, (200, 100), (440, 500), (255, 255, 255), -1)
            cv2.circle(test_image, (320, 80), 30, (255, 255, 255), -1)  # 머리
            print(f"✓ Created synthetic test image with person shape")
        
        print(f"Test image shape: {test_image.shape}")
        
        # 3. 모델 추론 테스트
        print("\n=== INFERENCE TEST ===")
        
        # 다양한 신뢰도 임계값으로 테스트
        for conf_threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
            print(f"\nTesting with conf_threshold={conf_threshold}")
            
            results = model(test_image, verbose=False, conf=conf_threshold)
            
            print(f"Number of detections: {len(results[0].boxes)}")
            
            for i, box in enumerate(results[0].boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id] if cls_id < len(model.names) else f"unknown_{cls_id}"
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                print(f"  Detection {i}: class={class_name}, conf={conf:.3f}, bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
        
        # 4. 모델 정보 출력
        print(f"\n=== MODEL INFO ===")
        print(f"Model info: {model.info()}")
        
        # 5. 디바이스 확인
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # bestcctv.pt 모델 디버깅
    success = debug_model("models/weights/bestcctv.pt")
    
    if success:
        print("\n✓ Model debugging completed successfully")
    else:
        print("\n✗ Model debugging failed") 