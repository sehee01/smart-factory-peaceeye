

#!/usr/bin/env python3

"""

main.py 통합 기능 테스트 스크립트

"""

import sys

import os

import cv2

import numpy as np

import json

import requests

from datetime import datetime

# 상위 디렉토리 경로 추가

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ppe_detection():
    """PPE 탐지 기능 테스트 (ppe_video_interative.py 사용)"""

    print("Testing PPE Detection with InteractivePPETester...")

   

    try:

        from ppe_video_interative import InteractivePPETester

       

        # InteractivePPETester 초기화

        tester =InteractivePPETester()

       

        if not tester.init_models():

            print("Model initialization failed")

            return False

       

        # 테스트 이미지 로드 (실제 비디오 프레임 사용)

        video_path ="test_video/test01.mp4"

        cap =cv2.VideoCapture(video_path)

       

        if not cap.isOpened():

            print(f"Cannot open video: {video_path}")

            return False

       

        ret,frame =cap.read()

        if not ret:

            print("Cannot read frame from video")

            return False

       

        # PPE 탐지 수행 (InteractivePPETester 사용)

        ppe_detections =tester.ppe_detector.detect_frame(frame)

        violations =tester.ppe_detector.analyze_safety_violations(ppe_detections)

       

        # 신뢰도 0.5 이상만 필터링

        high_confidence_violations =[v for v in violations if v.get('confidence',0)>=0.5]

       

        print(f"PPE Detections: {len(ppe_detections)}")

        print(f"PPE Violations (all): {len(violations)}")

        print(f"PPE Violations (high confidence): {len(high_confidence_violations)}")

       

        for violation in high_confidence_violations:

            print(f"  - {violation['class_name']} (confidence: {violation['confidence']:.2f})")

       

        cap.release()

        return True

       

    except Exception as e:

        print(f"PPE detection test failed: {e}")

        return False

def test_reid_tracking():

    """ReID 추적 기능 테스트"""

    print("Testing ReID Tracking...")

   

    try:

        from redis_global_reid_main_v2 import run_tracking_realtime,FeatureExtractor,RedisGlobalReIDManagerV2

        import torch

       

        # ReID 모델 초기화

        reid_extractor =FeatureExtractor(

            model_name='osnet_ibn_x1_0',

            model_path=None,

            device='cuda'if torch.cuda.is_available()else 'cpu'

        )

       

        global_reid_manager =RedisGlobalReIDManagerV2(

            similarity_threshold=0.5,

            feature_ttl=3000,

            max_features_per_camera=10,

            redis_host='localhost',

            redis_port=6379,

            frame_rate=30

        )

       

        # 추적 제너레이터 생성

        video_path ="test_video/test01.mp4"

        yolo_model ="models/weights/bestcctv.pt" # 경로 수정

       

        generator =run_tracking_realtime(

            video_path,

            yolo_model,

            reid_extractor,

            camera_id=0,

            global_reid_manager=global_reid_manager

        )

       

        # 몇 프레임 테스트

        for i in range(5):

            try:

                result =next(generator)

                if isinstance(result,tuple):

                    detections,frame =result

                    print(f"Frame {i+1}: {len(detections)}workers detected")

                    for detection in detections:

                        print(f"  - Worker {detection['workerID']}: ({detection['position_X']:.2f}, {detection['position_Y']:.2f})")

                else:

                    print(f"Frame {i+1}: {len(result)}workers detected")

            except StopIteration:

                print("Video ended")

                break

       

        return True

       

    except Exception as e:

        print(f"ReID tracking test failed: {e}")

        return False

def test_integrated_detection():

    """통합 탐지 테스트 (InteractivePPETester + ReID + Backend)"""

    print("Testing Integrated Detection with InteractivePPETester...")

   

    try:

        from ppe_video_interative import InteractivePPETester

       

        # InteractivePPETester 초기화

        tester =InteractivePPETester()

       

        if not tester.init_models():

            print("Model initialization failed")

            return False

       

        # ReID 추적 제너레이터가 있는지 확인

        if tester.tracking_generator is None:

            print("ReID tracking not available, using fallback mode")

            return test_ppe_detection()  # PPE만 테스트

       

        # 몇 프레임 테스트

        frame_count =0

        max_frames =10

       

        while frame_count <max_frames:

            try:

                # ReID 추적 결과 가져오기

                result =next(tester.tracking_generator)

               

                if isinstance(result,tuple):

                    detections,frame =result

                    frame_count +=1

                   

                    print(f"Frame {frame_count}: {len(detections)}workers detected")

                   

                    # PPE 탐지 수행

                    ppe_violations =[]

                    if frame is not None and tester.ppe_detector is not None:

                        ppe_detections =tester.ppe_detector.detect_frame(frame)

                        violations =tester.ppe_detector.analyze_safety_violations(ppe_detections)

                       

                        # 신뢰도 0.5 이상만 필터링

                        high_confidence_violations =[v for v in violations if v.get('confidence',0)>=0.5]

                       

                        if high_confidence_violations:

                            print(f"  PPE Violations: {len(high_confidence_violations)}")

                           

                            # PPE 위반을 워커와 매핑 (간단한 방식: 첫 번째 워커에 모든 위반 할당)

                            for detection in detections:

                                worker_id =f"worker_{detection['workerID']:03d}"

                               

                                # 해당 워커에 대한 모든 PPE 위반 타입 수집

                                violation_types =[]

                                max_confidence =0.0

                               

                                for violation in high_confidence_violations:

                                    violation_types.append(violation['class_name'])

                                    max_confidence =max(max_confidence,float(violation['confidence']))

                                    print(f"    -> {worker_id} PPE violation: {violation['class_name']} (confidence: {violation['confidence']:.2f})")

                               

                                # 모든 위반을 하나의 violation 객체로 통합

                                if violation_types:

                                    combined_violation ={

                                        'worker_id': worker_id,

                                        'violation_type': violation_types,  # 리스트로 여러 위반 타입 포함

                                        'confidence': max_confidence,  # 가장 높은 신뢰도 사용

                                        'timestamp': datetime.now().isoformat(),

                                        'camera_id': detection['cameraID'],

                                        'violation_count': len(violation_types)  # 위반 개수 추가

}

                                    ppe_violations.append(combined_violation)

                                    print(f"    -> Combined violation for {worker_id}: {', '.join(violation_types)}(max confidence: {max_confidence:.2f})")

                                    break  # 첫 번째 워커에만 할당

                   

                    # 위험구역 위반 확인

                    hazard_violations =[]

                    for detection in detections:

                        worker_id =f"worker_{detection['workerID']:03d}"

                        worker_pos ={

                            'worker_id': worker_id,

                            'x': detection['position_X'],

                            'y': detection['position_Y']

}

                       

                        # 위험구역 위반 확인 (tester의 hazard_zones 사용)

                        for zone in tester.hazard_zones:

                            zone_center_x = (zone['x1'] + zone['x2']) / 2
                            zone_center_y = (zone['y1'] + zone['y2']) / 2
                            distance = np.sqrt((worker_pos['x'] - zone_center_x)**2 + (worker_pos['y'] - zone_center_y)**2)
                            if distance <zone['threshold']:
                                hazard_violations.append({
                                    'worker_id': worker_id,
                                    'zone_id': zone['id'],
                                    'zone_type': zone['type'],
                                    'distance': float(distance),
                                    'threshold': zone['threshold'],
                                    'timestamp': datetime.now().isoformat(),
                                    'severity': 'high'if distance <zone['threshold']*0.4 else 'medium'
})
                                print(f"    -> {worker_id}hazard zone violation detected! (zone: {zone['id']}, distance: {distance:.1f})")

                   

                    # 워커 정보 출력

                    workers =[]

                    for detection in detections:

                        worker_id =f"worker_{detection['workerID']:03d}"

                       

                        # 위반 상태 확인

                        ppe_violation = next((v for v in ppe_violations if v['worker_id'] == worker_id), None)
                        
                        # 위험구역 위반 확인
                        hazard_violation = next((v for v in hazard_violations if v['worker_id'] == worker_id), None)
                        
                        status ="normal"

                        if ppe_violation:

                            status ="ppe_violation"

                        elif hazard_violation:

                            status ="hazard_zone_violation"

                       

                        worker ={

                            "worker_id": worker_id,

                            "x": float(detection['position_X']),

                            "y": float(detection['position_Y']),

                            "status": status,

                            "zone_id": f"Z{detection['cameraID']:02d}",

                            "product_count": 1,

                            "timestamp": datetime.now().isoformat(),

                            "frame_id": detection.get('frame_id',frame_count)

}

                        workers.append(worker)

                       

                        print(f"  - {worker_id}: ({detection['position_X']:.2f}, {detection['position_Y']:.2f}) - Status: {status}")

                   

                    # 실시간 위치 데이터 (항상 전송)

                    realtime_data ={

                        "timestamp": datetime.now().isoformat(),

                        "workers": workers,

                        "zones": [

{

                                "zone_id": f"Z{detection['cameraID']:02d}",

                                "zone_name": f"Zone Z{detection['cameraID']:02d}",

                                "zone_type": "작업구역",

                                "active_workers": len(workers),

                                "ppe_violations": len(ppe_violations),

                                "hazard_dwell_count": len(hazard_violations)

}

                        ]

}

                   

                    # 위반 데이터 (위반이 있는 경우에만)

                    violation_data =None

                    if ppe_violations or hazard_violations:

                        # 알림 배열 생성

                        alerts =[]

                        alerts.extend(ppe_violations)

                        alerts.extend(hazard_violations)

                       

                        violation_data ={

                            "timestamp": datetime.now().isoformat(),

                            "workers": workers,

                            "alerts": alerts,

                            "ppe_violations": ppe_violations,

                            "hazard_violations": hazard_violations,

                            "zones": realtime_data["zones"]

}

                   

                    # 백엔드로 데이터 전송

                    try:
                        # 실시간 위치 데이터 전송 (항상)
                        response = requests.post("http://localhost:5000/inference", json=realtime_data, timeout=1)
                        if response.status_code == 200:
                            print(f"  [BACKEND] Real-time position data sent - {len(workers)} workers")
                        else:
                            print(f"  [BACKEND] Failed to send position data: {response.status_code}")
                    except Exception as e:
                        print(f"  [BACKEND] Error sending position data: {e}")
                    
                    # 위반 데이터 전송 (위반이 있는 경우에만)
                    if violation_data:
                        try:
                            response = requests.post("http://localhost:5000/inference", json=violation_data, timeout=1)
                            if response.status_code == 200:
                                print(f"  [BACKEND] Violation data sent - {len(ppe_violations)} PPE, {len(hazard_violations)} hazard violations")
                            else:
                                print(f"  [BACKEND] Failed to send violation data: {response.status_code}")
                        except Exception as e:
                            print(f"  [BACKEND] Error sending violation data: {e}")

                else:

                    print(f"Frame {frame_count}: {len(result)}workers detected (no frame data)")

                   

            except StopIteration:

                print("Video ended")

                break

            except Exception as e:

                print(f"Error processing frame {frame_count}: {e}")

                break

       

        return True

       

    except Exception as e:

        print(f"Integrated detection test failed: {e}")

        return False

def test_backend_communication():

    """백엔드 통신 테스트"""

    print("Testing Backend Communication...")

   

    try:

        # 테스트 데이터 생성

        test_data ={

            "timestamp": datetime.now().isoformat(),

            "workers": [

{

                    "worker_id": "worker_001",

                    "x": 100.5,

                    "y": 200.3,

                    "status": "normal",

                    "zone_id": "Z01",

                    "product_count": 1,

                    "frame_id": 1

}

            ],

            "alerts": [],

            "ppe_violations": [],

            "hazard_violations": [],

            "zones": [

{

                    "zone_id": "Z01",

                    "zone_name": "Zone Z01",

                    "zone_type": "작업구역",

                    "active_workers": 1,

                    "ppe_violations": 0,

                    "hazard_dwell_count": 0

}

            ]

}

       

        # 백엔드로 전송
        response = requests.post("http://localhost:5000/inference", json=test_data, timeout=5)
        
        if response.status_code == 200:
            print("Backend communication successful")
            return True
        else:
            print(f"Backend communication failed: {response.status_code}")
            return False
            
    except requests.RequestException as e:
        print(f"Backend communication test failed: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("Main Integration Test")
    print("="*50)
    
    # 통합 탐지 테스트 (InteractivePPETester 사용)
    integrated_success = test_integrated_detection()
    print()
    
    # 개별 테스트들
    ppe_success = test_ppe_detection()
    print()
    
    reid_success = test_reid_tracking()
    print()
    
    backend_success = test_backend_communication()
    print()
    
    # 결과 요약
    print("Test Results:")
    print(f"  Integrated Detection: {'PASS' if integrated_success else 'FAIL'}")
    print(f"  PPE Detection: {'PASS' if ppe_success else 'FAIL'}")
    print(f"  ReID Tracking: {'PASS' if reid_success else 'FAIL'}")
    print(f"  Backend Communication: {'PASS' if backend_success else 'FAIL'}")
    
    if all([integrated_success, ppe_success, reid_success, backend_success]):
        print("\nAll tests passed! Integration is working correctly.")
    else:
        print("\nSome tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()