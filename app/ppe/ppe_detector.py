import cv2
import numpy as np


class PPEDetector:
    """PPE 탐지 클래스"""
    
    def __init__(self, model_path="models/weights/best_yolo11n.pt"):
        self.model_path = model_path
        self.model = None
        self.class_names = []
        self.ppe_classes = {
            'helmet': ['helmet', 'hardhat', 'safety_helmet'],
            'vest': ['vest', 'safety_vest', 'reflective_vest'],
            'gloves': ['gloves', 'safety_gloves'],
            'safety_shoes': ['safety_shoes', 'steel_toe_shoes'],
            'goggles': ['goggles', 'safety_goggles', 'eye_protection']
        }
        self.required_ppe = ['helmet', 'vest']  # 필수 PPE 항목
        self.init_model()
    
    def init_model(self):
        """PPE 탐지 모델 초기화"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            
            # 클래스 이름 로드 (모델에서 가져오거나 기본값 사용)
            if hasattr(self.model, 'names'):
                self.class_names = list(self.model.names.values())
            else:
                # 기본 PPE 클래스 이름
                self.class_names = [
                    'helmet', 'vest', 'gloves', 'safety_shoes', 'goggles',
                    'no_helmet', 'no_vest', 'no_gloves', 'no_safety_shoes', 'no_goggles'
                ]
            
            print(f"PPE Detection Model loaded: {self.model_path}")
            print(f"PPE Classes: {self.class_names}")
            
        except Exception as e:
            print(f"Failed to load PPE detection model: {e}")
            self.model = None
    
    def detect_ppe_violations(self, frame, person_bbox):
        """특정 사람의 PPE 위반 탐지"""
        if self.model is None:
            return []
        
        try:
            # 사람 영역 크롭
            x1, y1, x2, y2 = map(int, person_bbox)
            person_crop = frame[y1:y2, x1:x2]
            
            if person_crop.size == 0:
                return []
            
            # PPE 탐지
            results = self.model(person_crop, verbose=False)
            detections = results[0]
            
            violations = []
            detected_ppe = set()
            
            if detections.boxes is not None:
                boxes = detections.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if cls_id < len(self.class_names):
                        class_name = self.class_names[cls_id].lower()
                        
                        # PPE 항목 확인
                        for ppe_type, ppe_names in self.ppe_classes.items():
                            if class_name in ppe_names:
                                detected_ppe.add(ppe_type)
                                break
                        
                        # 위반 항목 확인 (no_ 접두사)
                        if class_name.startswith('no_'):
                            violation_type = class_name[3:]  # 'no_' 제거
                            violations.append({
                                'type': violation_type,
                                'confidence': conf,
                                'class_name': class_name
                            })
            
            # 필수 PPE 누락 확인
            for required in self.required_ppe:
                if required not in detected_ppe:
                    violations.append({
                        'type': required,
                        'confidence': 1.0,
                        'class_name': f'no_{required}',
                        'missing': True
                    })
            
            return violations
            
        except Exception as e:
            print(f"PPE detection error: {e}")
            return []

