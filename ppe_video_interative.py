#!/usr/bin/env python3

"""

test.mp4 영상으로 인터랙티브 PPE 및 위험구역 탐지 테스트 스크립트 (ReID 통합 버전)

"""

import sys

import os

import cv2

import numpy as np

import json

import requests

import asyncio

import torch

from datetime import datetime

from PIL import Image,ImageDraw,ImageFont

import io

# 상위 디렉토리 경로 추가

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'ByteTrack'))

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'deep-person-reid-master'))

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'models','mapping'))

# Node.js 서버의 POST 수신 URL

NODE_SERVER_URL ="http://localhost:5000/inference"

class InteractivePPETester:

    def __init__(self):

        self.video_path ="test_video/test01.mp4"

        self.hazard_zones =[] # 위험구역 리스트

        self.drawing_zone =False

        self.start_point =None

        self.end_point =None

        self.zone_id_counter =1

       

        # ReID 추적 관련 변수

        self.reid_extractor =None

        self.global_reid_manager =None

        self.tracking_generator =None

       

        # 위반 이력

        self.ppe_violation_history ={}

        self.hazard_violation_history ={}

       

        # 모델 초기화

        self.init_models()

       

    def init_models(self):

        """Initialize all models (PPE, ReID, etc.)"""

        try:

            # PPE 탐지 모듈 초기화

            from ppe_detector import PPEDetector,ViolationHistory

           

            model_path ="models/weights/best_yolo11n.pt"

            if not os.path.exists(model_path):

                print(f"PPE model file not found: {model_path}")

                return False

           

            detection_items ={

                'detect_no_safety_vest_or_helmet': True,

                'detect_near_machinery_or_vehicle': True,

                'detect_in_restricted_area': False

}

           

            self.ppe_detector =PPEDetector(model_path,detection_items)

            self.violation_history =ViolationHistory()

            print("PPE detection model initialization successful")

           

            # ReID 모델 초기화

            try:

                from redis_global_reid_main_v2 import FeatureExtractor, RedisGlobalReIDManagerV2, run_tracking_realtime

               

                self.reid_extractor =FeatureExtractor(

                    model_name='osnet_ibn_x1_0',

                    model_path=None,

                    device='cuda'if torch.cuda.is_available()else 'cpu'

                )

               

                self.global_reid_manager =RedisGlobalReIDManagerV2(

                    similarity_threshold=0.5,

                    feature_ttl=3000,

                    max_features_per_camera=10,

                    redis_host='localhost',

                    redis_port=6379,

                    frame_rate=30

                )

               

                # 추적 제너레이터 생성

                yolo_model ="models/weights/bestcctv.pt"  # 경로 수정

                self.tracking_generator =run_tracking_realtime(

                    self.video_path,

                    yolo_model,

                    self.reid_extractor,

                    camera_id=0,

                    global_reid_manager=self.global_reid_manager

                )

               

                print("ReID tracking model initialization successful")

               

            except Exception as e:

                print(f"ReID model initialization failed: {e}")

                # ReID 실패해도 PPE는 계속 사용

                self.reid_extractor =None

                self.global_reid_manager =None

                self.tracking_generator =None

           

            return True

           

        except Exception as e:

            print(f"Model initialization failed: {e}")

            return False

   

    def mouse_callback(self, event, x, y, flags, param):

        """Mouse callback function - hazard zone setting"""

        if event ==cv2.EVENT_LBUTTONDOWN:

            if not self.drawing_zone:

                # Start drawing new zone

                self.drawing_zone =True

                self.start_point =(x,y)

                print(f"Hazard zone {self.zone_id_counter}start point set: ({x}, {y})")

            else:

                # Complete zone drawing

                self.drawing_zone =False

                self.end_point =(x,y)

               

                # Add hazard zone

                zone = {
                    'id': f"zone_{self.zone_id_counter}",
                    'x1': min(self.start_point[0], self.end_point[0]),
                    'y1': min(self.start_point[1], self.end_point[1]),
                    'x2': max(self.start_point[0], self.end_point[0]),
                    'y2': max(self.start_point[1], self.end_point[1]),
                    'type': 'hazard',
                    'threshold': 50
                }
                
                self.hazard_zones.append(zone)
                self.zone_id_counter += 1
                print(f"Hazard zone {self.zone_id_counter} setup complete: ({zone['x1']}, {zone['y1']}) ~ ({zone['x2']}, {zone['y2']})")
                
                # Reset points
                self.start_point = None
                self.end_point = None
                self.drawing_zone = False

   

    def check_hazard_zone_violation(self, worker_position):

        """Check if worker is approaching hazard zone"""

        violations =[]

       

        for zone in self.hazard_zones:

            # Calculate distance to zone center
            zone_center_x = (zone['x1'] + zone['x2']) / 2
            zone_center_y = (zone['y1'] + zone['y2']) / 2
            
            # Calculate distance between worker and zone center
            distance = np.sqrt((worker_position['x'] - zone_center_x)**2 + (worker_position['y'] - zone_center_y)**2)
            
            if distance < zone['threshold']:
                violations.append({
                    'worker_id': worker_position['worker_id'],
                    'zone_id': zone['id'],
                    'zone_type': zone['type'],
                    'distance': float(distance),
                    'threshold': zone['threshold'],
                    'timestamp': datetime.now().isoformat(),
                    'severity': 'high' if distance < zone['threshold'] * 0.4 else 'medium'
                })
        
        return violations

   

    def draw_hazard_zones(self, frame):

        """Draw hazard zones on frame"""

        for zone in self.hazard_zones:

            # Draw zone rectangle

            cv2.rectangle(frame,(zone['x1'],zone['y1']),(zone['x2'],zone['y2']),(0,0,255),2)

           

            # Display zone ID

            cv2.putText(frame,zone['id'],(zone['x1'],zone['y1']-10),

                       cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

           

            # Draw hazard range circle

            center_x =int((zone['x1']+zone['x2'])/2)

            center_y =int((zone['y1']+zone['y2'])/2)

            cv2.circle(frame,(center_x,center_y),zone['threshold'],(0,255,255),1)

   

    def draw_current_zone(self, frame):

        """Display currently drawing zone"""

        if self.drawing_zone and self.start_point:

            cv2.circle(frame,self.start_point,5,(255,0,0),-1)

            cv2.putText(frame,f"Zone {self.zone_id_counter}Start",

                       (self.start_point[0]+10,self.start_point[1]-10),

                       cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

   

    def put_korean_text(self, frame, text, position, font_size=0.6, color=(255, 255, 255), thickness=2):

        """한글 텍스트를 이미지에 그리기"""

        try:

            # PIL을 사용하여 한글 텍스트 그리기

            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            draw = ImageDraw.Draw(pil_img)

            

            # 폰트 설정 - 여러 폰트 시도

            font = None

            font_paths = [

                "arial.ttf",

                "C:/Windows/Fonts/arial.ttf",

                "C:/Windows/Fonts/malgun.ttf",

                "C:/Windows/Fonts/gulim.ttc",

                "C:/Windows/Fonts/batang.ttc"

            ]

            

            for font_path in font_paths:

                try:

                    font = ImageFont.truetype(font_path, int(font_size * 20))

                    break

                except:

                    continue

            

            if font is None:

                font = ImageFont.load_default()

            

            # 텍스트 그리기

            draw.text(position, text, font=font, fill=color[::-1])  # RGB to BGR

            

            # PIL 이미지를 OpenCV 형식으로 변환

            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        except Exception as e:

            # 한글 표시 실패 시 영어로 대체

            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)

            return frame

   

    def draw_instructions(self, frame):

        """Display usage instructions"""

        instructions = [

            "Usage:",

            "1. Click mouse to set hazard zone",

            "2. First click: zone start point",

            "3. Second click: zone end point",

            "4. 'q': quit, 'r': reset zones, 's': save zones"

        ]

        

        y_offset = 30

        for instruction in instructions:

            cv2.putText(frame, instruction, (10, y_offset),

                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            y_offset += 25

   

    def run_interactive_test(self):

        """Run interactive test"""

        print("Interactive PPE and Hazard Zone Detection Test Started...")

        print("Usage:")

        print("   - Click mouse to set hazard zone")

        print("   - First click: zone start point")

        print("   - Second click: zone end point")

        print("   - 'q': quit, 'r': reset zones, 's': save zones")

       

        # Initialize video capture

        cap =cv2.VideoCapture(self.video_path)

        if not cap.isOpened():

            print(f"Cannot open video file: {self.video_path}")

            return False

       

        # Create window and set mouse callback

        window_name ="PPE & Hazard Zone Detection Test"

        cv2.namedWindow(window_name)

        cv2.setMouseCallback(window_name,self.mouse_callback)

       

        frame_count =0

        total_ppe_violations =0

        total_hazard_violations =0

       

        while True:

            ret,frame =cap.read()

            if not ret:

                # Restart video from beginning when it ends

                cap.set(cv2.CAP_PROP_POS_FRAMES,0)

                continue

           

            frame_count +=1

           

            # Create frame copy (preserve original)

            display_frame =frame.copy()

           

            # Perform PPE detection every 10 frames (performance optimization)
            if frame_count % 10 == 0:
                try:
                    # Perform PPE detection
                    ppe_detections = self.ppe_detector.detect_frame(frame)
                    ppe_violations = self.ppe_detector.analyze_safety_violations(ppe_detections, frame_count)
                    
                    if ppe_violations:
                        # Filter violations with confidence >= 0.5
                        high_confidence_violations = [v for v in ppe_violations if v.get('confidence', 0) >= 0.5]
                        
                        if high_confidence_violations:
                            total_ppe_violations += len(high_confidence_violations)
                            print(f"Frame {frame_count}: {len(high_confidence_violations)} PPE violations detected (confidence >= 0.5)")
                            
                            # Assign PPE violations to workers (use ReID results in practice)
                            for i, violation in enumerate(high_confidence_violations):
                                worker_id = f"worker_{i+1:03d}"
                                print(f"   - {worker_id}: {violation['class_name']} (confidence: {violation['confidence']:.2f})")
                                
                                # Update worker position
                                bbox = violation['bbox']
                                self.workers[worker_id]['x'] = int((bbox[0] + bbox[2]) / 2)
                                self.workers[worker_id]['y'] = int((bbox[1] + bbox[3]) / 2)
                                self.workers[worker_id]['last_seen'] = frame_count

                                # Add to violation history
                                self.violation_history.add_violation(
                                    violation['class_name'],
                                    frame_count,
                                    datetime.now().isoformat(),
                                    violation.get('bbox', [0, 0, 0, 0]),
                                    violation['confidence'],
                                    f"{worker_id}- detected in frame {frame_count}"
                                )
                    
                    # Check hazard zone violations (for active workers)
                    if self.hazard_zones:
                        for worker_id, worker_data in self.workers.items():
                            # Only check workers detected in last 30 frames
                            if frame_count - worker_data['last_seen'] < 30:
                                worker_pos = {
                                    'worker_id': worker_id,
                                    'x': worker_data['x'],
                                    'y': worker_data['y']
                                }
                                
                                hazard_violations = self.check_hazard_zone_violation(worker_pos)
                                if hazard_violations:
                                    total_hazard_violations += len(hazard_violations)
                                    print(f"Frame {frame_count}: {worker_id} approaching hazard zone")
                                    
                                    for violation in hazard_violations:
                                        print(f"   - {violation['worker_id']} accessing {violation['zone_id']} (distance: {violation['distance']:.1f})")
                                        
                                        # Add to violation history
                                        self.violation_history.add_violation(
                                            violation['worker_id'],
                                            'hazard_zone',
                                            violation['zone_id'],
                                            frame_count,
                                            violation['distance']
                                        )
                except Exception as e:
                    print(f"Error in PPE detection: {e}")
                    continue

            # Draw hazard zones
            for zone in self.hazard_zones:
                cv2.rectangle(frame, (zone['x1'], zone['y1']), (zone['x2'], zone['y2']), (0, 0, 255), 2)
                
                # Draw zone ID
                cv2.putText(frame, zone['id'], (zone['x1'], zone['y1'] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Draw zone center
                center_x = int((zone['x1'] + zone['x2']) / 2)
                center_y = int((zone['y1'] + zone['y2']) / 2)
                cv2.circle(frame, (center_x, center_y), 3, (0, 255, 0), -1)

            # Draw current zone being created
            if self.start_point and self.end_point:
                cv2.rectangle(display_frame, self.start_point, self.end_point, (0, 255, 0), 2)
                cv2.putText(display_frame, "Drawing zone...", 
                       (self.start_point[0] + 10, self.start_point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            self.draw_current_zone(display_frame)

            self.draw_instructions(display_frame)

           

            # Display statistics

            stats_text =[

                f"Frame: {frame_count}",

                f"Hazard Zones: {len(self.hazard_zones)}",

                f"PPE Violations: {total_ppe_violations}",

                f"Zone Violations: {total_hazard_violations}"

            ]

           

            y_offset =frame.shape[0]-100

            for stat in stats_text:

                cv2.putText(display_frame,stat,(10,y_offset),

                           cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

                y_offset +=20

           

            # Display frame

            cv2.imshow(window_name,display_frame)

           

            # Handle key input

            key =cv2.waitKey(30)&0xFF

            if key ==ord('q'):

                break

            elif key ==ord('r'):

                # Reset zones

                self.hazard_zones =[]

                self.zone_id_counter =1

                print("Hazard zones reset complete")

            elif key ==ord('s'):

                # Save zone information

                self.save_zones()

       

        # Cleanup

        cap.release()

        cv2.destroyAllWindows()

       

        # Final results output

        print("\n"+"="*50)

        print("Final Test Results")

        print("="*50)

        print(f"Total frames processed: {frame_count}")

        print(f"Hazard zones set: {len(self.hazard_zones)}")

        print(f"Total PPE violations: {total_ppe_violations}")

        print(f"Total zone violations: {total_hazard_violations}")

       

        # Check violation history

        all_violations =self.violation_history.get_all_violations()

        if all_violations:

            print(f"\nViolation history: {len(all_violations)}")

            for i,violation in enumerate(all_violations[-3:],1):  # Show only last 3

                print(f"   {i}. {violation['class_name']} - Frame {violation['frame_id']}")

       

        return True

   

    def save_zones(self):

        """Save hazard zone information to file"""

        try:

            zones_data ={

                'timestamp': datetime.now().isoformat(),

                'zones': self.hazard_zones

}

           

            with open('hazard_zones.json','w',encoding='utf-8')as f:

                json.dump(zones_data,f,indent=2,ensure_ascii=False)

           

            print(f"Hazard zone information saved: hazard_zones.json")

           

        except Exception as e:

            print(f"Failed to save hazard zones: {e}")

def main():

    """Main function"""

    print("Interactive PPE and Hazard Zone Detection Test")

    print("="*60)

   

    tester =InteractivePPETester()

   

    if not tester.init_models():

        print("Model initialization failed")

        return False

   

    try:

        tester.run_interactive_test()

        print("\nTest completed!")

        return True

       

    except KeyboardInterrupt:

        print("\nInterrupted by user")

        return False

    except Exception as e:

        print(f"\nError during test execution: {e}")

        return False

if __name__=="__main__":

    main()