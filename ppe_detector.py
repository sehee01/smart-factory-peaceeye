#!/usr/bin/env python3

"""

PPE (Personal Protective Equipment) Detection Demo

실시간으로 테스트 비디오에서 안전장비(안전모, 안전조끼 등)를 감지하는 데모 프로그램

"""

import cv2

import numpy as np

import time

import sys

import os

import json

from datetime import datetime

from typing import List,Dict,Tuple,Optional

from pathlib import Path

# YOLO 모델 경로 설정

MODEL_PATH = "models/weights/best_yolo11n.pt"

# 클래스 라벨 정의 (PPE 감지용)

CLASS_NAMES ={

    0: 'Hardhat',           # 안전모

    1: 'Mask',              # 마스크

    2: 'NO-Hardhat',        # 안전모 미착용

    3: 'NO-Mask',           # 마스크 미착용

    4: 'NO-Safety Vest',    # 안전조끼 미착용

    5: 'Person',            # 사람

    6: 'Safety Cone',       # 안전콘

    7: 'Safety Vest',       # 안전조끼

    8: 'Machinery',         # 기계

    9: 'Vehicle'            # 차량

}

# 색상 정의 (BGR 형식)

COLORS ={

    'Hardhat': (0,255,0),      # 녹색 - 안전모 착용

    'Safety Vest': (0,255,0), # 녹색 - 안전조끼 착용

    'NO-Hardhat': (0,0,255),   # 빨간색 - 안전모 미착용

    'NO-Safety Vest': (0,0,255),# 빨간색 - 안전조끼 미착용

    'Person': (255,255,0),     # 청록색 - 사람

    'Safety Cone': (255,0,255),# 자홍색 - 안전콘

    'Machinery': (255,0,0),    # 파란색 - 기계

    'Vehicle': (255,0,0),      # 파란색 - 차량

    'Mask': (0,255,255),       # 노란색 - 마스크

    'NO-Mask': (0,0,255)       # 빨간색 - 마스크 미착용

}

class ViolationHistory:

    """

위반 이력을 관리하는 클래스

"""

   

    def __init__(self):

        self.violations =[]

        self.violation_history_file ="violation_history.json"

   

    def add_violation(self, violation_type: str, frame_number: int, timestamp: str,

                     bbox: List[float], confidence: float, details: str =""):

        """

위반 기록을 추가합니다.

       

Args:

violation_type (str): 위반 유형 ('NO-Hardhat', 'NO-Safety Vest')

frame_number (int): 프레임 번호

timestamp (str): 타임스탬프

bbox (List[float]): 바운딩 박스 좌표

confidence (float): 신뢰도

details (str): 추가 세부사항

"""

        violation ={

            'type': violation_type,

            'frame_number': frame_number,

            'timestamp': timestamp,

            'bbox': bbox,

            'confidence': confidence,

            'details': details

}

        self.violations.append(violation)

   

    def get_violations_by_type(self, violation_type: str) -> List[Dict]:

        """

특정 유형의 위반 기록을 반환합니다.

       

Args:

violation_type (str): 위반 유형

           

Returns:

List[Dict]: 위반 기록 리스트

"""

        return [v for v in self.violations if v['type']==violation_type]

   

    def get_all_violations(self) -> List[Dict]:

        """

모든 위반 기록을 반환합니다.

       

Returns:

List[Dict]: 모든 위반 기록

"""

        return self.violations

   

    def save_to_file(self):

        """

위반 이력을 파일에 저장합니다.

"""

        try:

            with open(self.violation_history_file,'w',encoding='utf-8')as f:

                json.dump(self.violations,f,ensure_ascii=False,indent=2)

            print(f"✅ 위반 이력이 {self.violation_history_file}에 저장되었습니다.")

        except Exception as e:

            print(f"❌ 위반 이력 저장 실패: {e}")

   

    def load_from_file(self):

        """

파일에서 위반 이력을 로드합니다.

"""

        try:

            if os.path.exists(self.violation_history_file):

                with open(self.violation_history_file,'r',encoding='utf-8')as f:

                    self.violations =json.load(f)

                print(f"✅ 위반 이력을 {self.violation_history_file}에서 로드했습니다.")

        except Exception as e:

            print(f"❌ 위반 이력 로드 실패: {e}")

   

    def print_violation_summary(self):

        """

위반 이력 요약을 출력합니다.

"""

        if not self.violations:

            print("📋 위반 기록이 없습니다.")

            return

       

        print("\n"+"="*60)

        print("📋 위반 이력 요약")

        print("="*60)

       

        # 위반 유형별 통계

        violation_counts ={}

        for violation in self.violations:

            v_type =violation['type']

            violation_counts[v_type]=violation_counts.get(v_type,0)+1

       

        print("위반 유형별 통계:")

        for v_type,count in violation_counts.items():

            print(f"  - {v_type}: {count}건")

       

        print(f"\n총 위반 건수: {len(self.violations)}건")

       

        # 최근 위반 10건 출력

        print(f"\n최근 위반 10건:")

        print("-"*60)

        recent_violations = sorted(self.violations, key=lambda x: x['timestamp'], reverse=True)[:10]

       

        for i,violation in enumerate(recent_violations,1):

            print(f"{i:2d}. [{violation['timestamp']}] {violation['type']}"

                  f"(프레임: {violation['frame_number']}, 신뢰도: {violation['confidence']:.2f})")

   

    def print_detailed_violations(self, violation_type: str =None):

        """

상세한 위반 이력을 출력합니다.

       

Args:

violation_type (str, optional): 특정 위반 유형만 출력

"""

        violations_to_show =self.violations

        if violation_type:

            violations_to_show =self.get_violations_by_type(violation_type)

       

        if not violations_to_show:

            print(f"📋 {violation_type or '위반'}기록이 없습니다.")

            return

       

        print(f"\n"+"="*80)

        print(f"📋 상세 위반 이력 {f'({violation_type})'if violation_type else ''}")

        print("="*80)

       

        for i,violation in enumerate(violations_to_show,1):

            print(f"\n{i:3d}. 위반 유형: {violation['type']}")

            print(f"    시간: {violation['timestamp']}")

            print(f"    프레임: {violation['frame_number']}")

            print(f"    신뢰도: {violation['confidence']:.3f}")

            print(f"    위치: {violation['bbox']}")

            if violation['details']:

                print(f"    세부사항: {violation['details']}")

class PPEDetector:

    """

A class to detect Personal Protective Equipment (PPE) in video frames.

"""

    def __init__(self, model_path: str, detection_items: Dict[str,bool]={}):

        """

Initialises the PPE detector.

Args:

model_path (str): The path to the YOLO model file.

detection_items (Dict[str, bool]): A dictionary of detection items

to enable/disable specific safety checks. The keys are:

- 'detect_no_safety_vest_or_helmet': Detect if workers are not

wearing hardhats or safety vests.

- 'detect_near_machinery_or_vehicle': Detect if workers are

dangerously close to machinery or vehicles.

- 'detect_in_restricted_area': Detect if workers are entering

restricted areas.

Raises:

RuntimeError: If the model file cannot be loaded.

"""

        self.model_path =model_path

        self.detection_items =detection_items

        self.violation_history =ViolationHistory()

       

        # Define required keys

        required_keys ={

            'detect_no_safety_vest_or_helmet',

            'detect_near_machinery_or_vehicle',

            'detect_in_restricted_area',

}

        # Validate detection_items type and content

        if isinstance(detection_items,dict)and all(

            isinstance(k,str)and isinstance(v,bool)

            for k,v in detection_items.items()

        )and required_keys.issubset(detection_items.keys()):

            self.detection_items =detection_items

        else:

            self.detection_items ={}

        # Load YOLO model

        self._load_model()

    def _load_model(self) -> None:

        """

Loads the YOLO model from the specified path.

Raises:

RuntimeError: If the model cannot be loaded.

"""

        try:

            from ultralytics import YOLO

            self.model =YOLO(self.model_path)

            print(f"✅ Model loaded successfully: {self.model_path}")

        except ImportError:

            raise RuntimeError(

                "ultralytics package is not installed. "

                "Please install it using: pip install ultralytics"

            )

        except Exception as e:

            raise RuntimeError(f"Failed to load model: {e}")

    def detect_frame(

        self,

        frame: np.ndarray,

) -> List[Dict[str,any]]:

        """

Detects PPE items in a single frame.

Args:

frame (np.ndarray): The input frame to process.

Returns:

List[Dict[str, any]]: List of detections with bbox, confidence,

class_id, and class_name.

"""

        try:

            # Perform detection using YOLO model

            results =self.model(frame,verbose=False)

           

            detections =[]

            for result in results:

                boxes =result.boxes

                if boxes is not None:

                    for box in boxes:

                        # Bounding box coordinates

                        x1,y1,x2,y2 =box.xyxy[0].cpu().numpy()

                        # Confidence score

                        confidence =box.conf[0].cpu().numpy()

                        # Class ID

                        class_id =int(box.cls[0].cpu().numpy())

                       

                        detections.append({

                            'bbox': [x1,y1,x2,y2],

                            'confidence': confidence,

                            'class_id': class_id,

                            'class_name': CLASS_NAMES.get(class_id,f'Unknown-{class_id}')

})

           

            return detections

        except Exception as e:

            print(f"Error during detection: {e}")

            return []

    def detect_danger(

        self,

        datas: List[List[float]],

) -> Tuple[List[str],List]:

        """

Detects potential safety violations in a construction site.

This function checks for safety violations:

1. Workers not wearing hardhats or safety vests.

2. Workers dangerously close to machinery or vehicles.

Args:

datas (List[List[float]]): A list of detections which includes

bounding box coordinates, confidence score, and class label.

Returns:

Tuple[List[str], List]: Warnings and additional data.

"""

        # Initialize the list to store warning messages

        warnings: set[str]=set()

        # Normalize data (convert to danger_detector format)

        normalized_data =self._normalize_detection_data(datas)

        ############################################################

        # Classify detected objects into different categories

        ############################################################

        # Persons

        persons = [d for d in normalized_data if d[5] == 5]
        hardhat_violations = [d for d in normalized_data if d[5] == 2]
        safety_vest_violations = [d for d in normalized_data if d[5] == 4]
        machinery_vehicles = [d for d in normalized_data if d[5] in [8, 9]]

        ############################################################

        # Check if people are not wearing hardhats or safety vests

        ############################################################

        if (

            not self.detection_items or

            self.detection_items.get('detect_no_safety_vest_or_helmet',False)

        ):

            self._check_safety_violations(

                persons,hardhat_violations,

                safety_vest_violations,warnings,

            )

        ############################################################

        # Check if people are dangerously close to machinery or vehicles

        ############################################################

        if (

            not self.detection_items or

            self.detection_items.get('detect_near_machinery_or_vehicle',False)

        ):

            self._check_proximity_violations(

                persons,machinery_vehicles,warnings,

            )

        return list(warnings),[]

    def _normalize_detection_data(self, detections: List[Dict]) -> List[List[float]]:

        """

Normalizes detection data to match danger_detector format.

Args:

detections (List[Dict]): Raw detection data.

Returns:

List[List[float]]: Normalized data in [x1, y1, x2, y2, conf, class_id] format.

"""

        normalized =[]

        for detection in detections:

            bbox =detection['bbox']

            confidence =detection['confidence']

            class_id =detection['class_id']

           

            # Convert to [x1, y1, x2, y2, confidence, class_id] format

            normalized.append([

                bbox[0], bbox[1], bbox[2], bbox[3],  # x1, y1, x2, y2

                confidence,                           # confidence

                class_id                             # class_id

            ])

       

        return normalized

    def _check_safety_violations(

        self,

        persons: List[List[float]],

        hardhat_violations: List[List[float]],

        safety_vest_violations: List[List[float]],

        warnings: set[str],

) -> None:

        """

Check for hardhat and safety vest violations.

Args:

persons (List[List[float]]): A list of person detections.

hardhat_violations (List[List[float]]): A list of hardhat violations.

safety_vest_violations (List[List[float]]): A list of safety vest violations.

warnings (set[str]): A set to store warning messages.

"""

        for violation in hardhat_violations +safety_vest_violations:

            label = 'NO-Hardhat' if violation[5] == 2 else 'NO-Safety Vest'

            if not any(

                self._overlap_percentage(violation[:4], p[:4]) > 0.5

                for p in persons

            ):

                warning_msg =(

                    'Warning: Someone is not wearing a hardhat!'

                    if label =='NO-Hardhat'

                    else 'Warning: Someone is not wearing a safety vest!'

                )

                warnings.add(warning_msg)

    def _check_proximity_violations(

        self,

        persons: List[List[float]],

        machinery_vehicles: List[List[float]],

        warnings: set[str],

) -> None:

        """

Check if anyone is dangerously close to machinery or vehicles.

Args:

persons (List[List[float]]): A list of person detections.

machinery_vehicles (List[List[float]]): A list of machinery and vehicle detections.

warnings (set[str]): A set to store warning messages.

"""

        for person in persons:

            for mv in machinery_vehicles:

                label = 'machinery' if mv[5] == 8 else 'vehicle'

                if self._is_dangerously_close(person[:4], mv[:4], label):

                    warning_msg =f"Warning: Someone is too close to {label}!"

                    warnings.add(warning_msg)

                    break

    def _overlap_percentage(self, bbox1: List[float], bbox2: List[float]) -> float:

        """

Calculate the overlap percentage between two bounding boxes.

Args:

bbox1 (List[float]): The first bounding box [x1, y1, x2, y2].

bbox2 (List[float]): The second bounding box [x1, y1, x2, y2].

Returns:

float: The overlap percentage.

"""

        # Calculate the coordinates of the intersection rectangle

        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        # Calculate the area of the intersection rectangle

        overlap_area =max(0,x2 -x1)*max(0,y2 -y1)

        # Calculate the area of both bounding boxes

        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        # Calculate the overlap percentage

        return overlap_area /float(area1 +area2 -overlap_area)

    def _is_dangerously_close(

        self,

        person_bbox: List[float],

        object_bbox: List[float],

        label: str

) -> bool:

        """

Determine if a person is dangerously close to machinery or vehicles.

Args:

person_bbox (List[float]): Bounding box of person [x1, y1, x2, y2].

object_bbox (List[float]): Bounding box of machinery/vehicle [x1, y1, x2, y2].

label (str): Type of the object ('machinery' or 'vehicle').

Returns:

bool: True if dangerously close, False otherwise.

"""

        # Calculate centers

        person_center = [(person_bbox[0] + person_bbox[2]) / 2,
                         (person_bbox[1] + person_bbox[3]) / 2]
        
        object_center = [(object_bbox[0] + object_bbox[2]) / 2,
                         (object_bbox[1] + object_bbox[3]) / 2]

       

        # Calculate distance

        distance = np.sqrt((person_center[0] - object_center[0])**2 +
                          (person_center[1] - object_center[1])**2)

       

        # Define threshold based on object type

        threshold = 80 if label == 'machinery' else 100

       

        return distance <threshold

    def draw_detections(

        self,

        frame: np.ndarray,

        detections: List[Dict[str,any]]

) -> np.ndarray:

        """

Draw detection results on the frame.

Args:

frame (np.ndarray): The input frame.

detections (List[Dict[str, any]]): Detection results.

Returns:

np.ndarray: Frame with detections drawn.

"""

        for detection in detections:

            bbox =detection['bbox']

            confidence =detection['confidence']

            class_name =detection['class_name']

           

            # Bounding box coordinates

            x1,y1,x2,y2 =map(int,bbox)

           

            # Color selection

            color =COLORS.get(class_name,(128,128,128))

           

            # Draw bounding box

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

           

            # Label text

            label =f"{class_name}: {confidence:.2f}"

           

            # Calculate label background size

            (label_width,label_height),_=cv2.getTextSize(

                label,cv2.FONT_HERSHEY_SIMPLEX,0.6,2

            )

           

            # Draw label background

            cv2.rectangle(frame,

                         (x1,y1 -label_height -10),

                         (x1 +label_width,y1),

                         color,-1)

           

            # Draw label text

            cv2.putText(frame,label,(x1,y1 -5),

                       cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

       

        return frame

    def analyze_safety_violations(

        self,

        detections: List[Dict[str,any]],

        frame_number: int =0

) -> List[Dict[str,any]]:

        """

Analyze safety violations from detections and record violations.

Args:

detections (List[Dict[str, any]]): Detection results.

frame_number (int): Current frame number for violation recording.

Returns:

List[Dict[str, any]]: List of violation objects.

"""

        violations =[]

       

        # Categorize detections

        people = [d for d in detections if d.get('class_name') == 'Person']
        no_hardhat = [d for d in detections if d.get('class_name') == 'NO-Hardhat']
        no_safety_vest = [d for d in detections if d.get('class_name') == 'NO-Safety Vest']
        no_mask = [d for d in detections if d.get('class_name') == 'NO-Mask']
        machinery = [d for d in detections if d.get('class_name') == 'Machinery']
        vehicles = [d for d in detections if d.get('class_name') == 'Vehicle']

       

        # Record violations in history and create violation objects

        timestamp =datetime.now().strftime("%Y-%m-%d%H:%M:%S")

       

        # Record hardhat violations

        for detection in no_hardhat:

            try:

                bbox =detection.get('bbox',[0,0,0,0])

                confidence =detection.get('confidence',0.0)

               

                violation_obj ={

                    'class_name': 'NO-Hardhat',

                    'bbox': bbox,

                    'confidence': confidence,

                    'timestamp': timestamp,

                    'frame_number': frame_number

}

                violations.append(violation_obj)

               

                self.violation_history.add_violation(

                    'NO-Hardhat',

                    frame_number,

                    timestamp,

                    bbox,

                    confidence,

                    "안전모 미착용 위반"

                )

            except Exception as e:

                print(f"Error processing hardhat violation: {e}")

       

        # Record safety vest violations

        for detection in no_safety_vest:

            try:

                bbox =detection.get('bbox',[0,0,0,0])

                confidence =detection.get('confidence',0.0)

               

                violation_obj ={

                    'class_name': 'NO-Safety Vest',

                    'bbox': bbox,

                    'confidence': confidence,

                    'timestamp': timestamp,

                    'frame_number': frame_number

}

                violations.append(violation_obj)

               

                self.violation_history.add_violation(

                    'NO-Safety Vest',

                    frame_number,

                    timestamp,

                    bbox,

                    confidence,

                    "안전조끼 미착용 위반"

                )

            except Exception as e:

                print(f"Error processing safety vest violation: {e}")

       

        # Record mask violations

        for detection in no_mask:

            try:

                bbox =detection.get('bbox',[0,0,0,0])

                confidence =detection.get('confidence',0.0)

               

                violation_obj ={

                    'class_name': 'NO-Mask',

                    'bbox': bbox,

                    'confidence': confidence,

                    'timestamp': timestamp,

                    'frame_number': frame_number

}

                violations.append(violation_obj)

               

                self.violation_history.add_violation(

                    'NO-Mask',

                    frame_number,

                    timestamp,

                    bbox,

                    confidence,

                    "마스크 미착용 위반"

                )

            except Exception as e:

                print(f"Error processing mask violation: {e}")

       

        # Check proximity violations

        for person in people:

            try:

                person_bbox =person.get('bbox',[0,0,0,0])

                person_center =[(person_bbox[0]+person_bbox[2])/2,

                               (person_bbox[1]+person_bbox[3])/2]

               

                # Check machinery proximity

                for machine in machinery:

                    machine_bbox =machine.get('bbox',[0,0,0,0])

                    machine_center = [(machine_bbox[0] + machine_bbox[2]) / 2,
                                      (machine_bbox[1] + machine_bbox[3]) / 2]
        
                    distance = np.sqrt((person_center[0] - machine_center[0])**2 +
                                      (person_center[1] - machine_center[1])**2)

                    if distance <80:

                        violation_obj ={

                            'class_name': 'Near-Machinery',

                            'bbox': person_bbox,

                            'confidence': person.get('confidence',0.0),

                            'timestamp': timestamp,

                            'frame_number': frame_number,

                            'distance': distance

}

                        violations.append(violation_obj)

               

                # Check vehicle proximity

                for vehicle in vehicles:

                    vehicle_bbox =vehicle.get('bbox',[0,0,0,0])

                    vehicle_center = [(vehicle_bbox[0] + vehicle_bbox[2]) / 2,
                                      (vehicle_bbox[1] + vehicle_bbox[3]) / 2]
        
                    distance = np.sqrt((person_center[0] - vehicle_center[0])**2 +
                                      (person_center[1] - vehicle_center[1])**2)

                    if distance <100:

                        violation_obj ={

                            'class_name': 'Near-Vehicle',

                            'bbox': person_bbox,

                            'confidence': person.get('confidence',0.0),

                            'timestamp': timestamp,

                            'frame_number': frame_number,

                            'distance': distance

}

                        violations.append(violation_obj)

            except Exception as e:

                print(f"Error processing proximity violation: {e}")

       

        return violations

def main() -> None:

    """

Main function to demonstrate the usage of the PPEDetector class.

"""

    print("🚧 PPE (Personal Protective Equipment) Detection Demo")

    print("="*60)

   

    # Check model path

    if not os.path.exists(MODEL_PATH):

        print(f"❌ Model file not found: {MODEL_PATH}")

        print("Please check if the YOLO model file exists in models/pt/ directory.")

        return

   

    # Check video file path

    video_path ="test_video\KSEB03.mp4"

    if not os.path.exists(video_path):

        print(f"❌ Video file not found: {video_path}")

        return

   

    # Initialize PPE detector with default detection items

    print("🔧 Initializing PPE detector...")

    detection_items ={

        'detect_no_safety_vest_or_helmet': True,

        'detect_near_machinery_or_vehicle': True,

        'detect_in_restricted_area': False  # Not implemented in this demo

}

   

    detector =PPEDetector(MODEL_PATH,detection_items)

   

    # Load existing violation history

    detector.violation_history.load_from_file()

   

    # Open video capture

    print(f"📹 Opening video file: {video_path}")

    cap =cv2.VideoCapture(video_path)

   

    if not cap.isOpened():

        print("❌ Cannot open video file.")

        return

   

    # Get video information

    fps =cap.get(cv2.CAP_PROP_FPS)

    frame_count =int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width =int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height =int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

   

    print(f"📊 Video information:")

    print(f"   - FPS: {fps:.2f}")

    print(f"   - Total frames: {frame_count}")

    print(f"   - Resolution: {width}x{height}")

    print()

    print("🎬 Starting real-time detection (Press 'q' to quit, 'h' for violation history)")

    print("-"*60)

   

    frame_idx =0

    start_time =time.time()

   

    while True:

        ret,frame =cap.read()

        if not ret:

            print("📺 Video playback completed")

            break

       

        frame_idx +=1

       

        # Perform PPE detection

        detections =detector.detect_frame(frame)

       

        # Draw detection results on frame

        frame =detector.draw_detections(frame,detections)

       

        # Analyze safety violations and record them

        violations =detector.analyze_safety_violations(detections,frame_idx)

       

        # Draw information panel

        info_panel =np.zeros((200,frame.shape[1],3),dtype=np.uint8)

       

        # Frame information

        cv2.putText(info_panel,f"Frame: {frame_idx}/{frame_count}",(10,30),

                   cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

       

        # Detected objects count

        cv2.putText(info_panel,f"Detected Objects: {len(detections)}",(10,60),

                   cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

       

        # Display violations

        y_offset =90

        # Aggregate violations by type (violations is a list)
        violation_counts_panel = {}
        for v in violations:
            v_type = v.get('class_name', 'Unknown')
            violation_counts_panel[v_type] = violation_counts_panel.get(v_type, 0) + 1

        for violation_type,count in violation_counts_panel.items():
            if count >0:
                color =(0,0,255)if count >0 else (255,255,255)
                text =f"{violation_type.replace('_',' ').title()}: {count}"
                cv2.putText(info_panel,text,(10,y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
                y_offset +=25

       

        # Add information panel to frame

        frame =np.vstack([frame,info_panel])

       

        # Display window

        cv2.imshow('PPE Detection Demo',frame)

       

        # Handle key input

        key =cv2.waitKey(1)&0xFF

        if key ==ord('q'):

            print("⏹️ User requested exit.")

            break

        elif key ==ord('p'):  # Pause

            print("⏸️ Paused (Press any key to resume)")

            cv2.waitKey(0)

        elif key ==ord('h'):  # Show violation history

            print("\n📋 위반 이력 출력 중...")

            detector.violation_history.print_violation_summary()

            print("\n계속하려면 아무 키나 누르세요...")

            cv2.waitKey(0)

        elif key ==ord('d'):  # Show detailed violations

            print("\n📋 상세 위반 이력 출력 중...")

            detector.violation_history.print_detailed_violations()

            print("\n계속하려면 아무 키나 누르세요...")

            cv2.waitKey(0)

        elif key ==ord('1'):  # Show hardhat violations only

            print("\n📋 안전모 미착용 위반 이력 출력 중...")

            detector.violation_history.print_detailed_violations('NO-Hardhat')

            print("\n계속하려면 아무 키나 누르세요...")

            cv2.waitKey(0)

        elif key ==ord('2'):  # Show safety vest violations only

            print("\n📋 안전조끼 미착용 위반 이력 출력 중...")

            detector.violation_history.print_detailed_violations('NO-Safety Vest')

            print("\n계속하려면 아무 키나 누르세요...")

            cv2.waitKey(0)

       

        # FPS limiting (real-time playback)

        time.sleep(1/fps)

   

    # Cleanup

    cap.release()

    cv2.destroyAllWindows()

   

    # Save violation history

    detector.violation_history.save_to_file()

   

    # Print final violation summary

    detector.violation_history.print_violation_summary()

   

    # Calculate execution time

    end_time =time.time()

    total_time =end_time -start_time

    print(f"\n⏱️ Total execution time: {total_time:.2f}seconds")

    print("👋 PPE Detection Demo finished")

    print("\n키 설명:")

    print("  - 'q': 종료")

    print("  - 'p': 일시정지")

    print("  - 'h': 위반 이력 요약 출력")

    print("  - 'd': 상세 위반 이력 출력")

    print("  - '1': 안전모 미착용 위반만 출력")

    print("  - '2': 안전조끼 미착용 위반만 출력")

if __name__=='__main__':

    main()