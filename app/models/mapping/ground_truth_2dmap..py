import cv2
import numpy as np
import json
import os
import time
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional

class GroundTruth2DMap:
    """
    실제 지면 길이를 기반으로 한 2D 캔버스 맵 시스템
    영상에서 객체를 감지하고 실제 지면 좌표로 변환하여 2D 맵에 표시
    """
    
    def __init__(self, video_path: str, calibration_file: str, model_path: Optional[str] = None):
        self.video_path = video_path
        self.calibration_file = calibration_file
        self.model_path = model_path
        
        # 캘리브레이션 데이터
        self.homography_matrix = None
        self.real_points = None
        self.bev_size = None
        self.pixels_per_meter = None
        self.real_map_size = None
        
        # YOLO 모델
        self.model = None
        
        # 비디오 캡처
        self.cap = None
        
        # 추적된 객체 정보
        self.tracked_objects = {}
        
        # 캘리브레이션 로드
        self.load_calibration()
        
        # YOLO 모델 초기화
        self.init_yolo_model()
    
    def load_calibration(self):
        """캘리브레이션 파일 로드"""
        if not os.path.exists(self.calibration_file):
            print(f"Calibration file not found: {self.calibration_file}")
            return False
        
        try:
            with open(self.calibration_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.homography_matrix = np.array(data['homography_matrix'])
            self.real_points = data['real_points']
            self.bev_size = data['bev_size']
            self.pixels_per_meter = data['pixels_per_meter']
            self.real_map_size = data['real_map_size']
            
            print(f"Calibration loaded successfully:")
            print(f"  - Real map size: {self.real_map_size['width_meters']:.1f}m x {self.real_map_size['height_meters']:.1f}m")
            print(f"  - 2D map size: {self.bev_size[0]} x {self.bev_size[1]} pixels")
            print(f"  - Scale: 1 meter = {self.pixels_per_meter} pixels")
            
            return True
            
        except Exception as e:
            print(f"Failed to load calibration file: {e}")
            return False
    
    def init_yolo_model(self):
        """YOLO 모델 초기화"""
        try:
            if self.model_path and os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                print(f"YOLO model loaded: {self.model_path}")
            else:
                self.model = YOLO('yolov8n.pt')
                print("Using default YOLO model: yolov8n.pt")
            
            return True
        except Exception as e:
            print(f"Failed to initialize YOLO model: {e}")
            return False
    
    def detect_and_track_objects(self, frame):
        """YOLO로 객체 감지 및 추적 (사람 클래스 발 위치만)"""
        if self.model is None:
            return []
        
        results = self.model.track(frame, persist=True, conf=0.5, iou=0.5)
        
        detections = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
                # 사람 클래스만 처리 (클래스 5: person)
                if int(cls) == 5:  # 사람 클래스만
                    x1, y1, x2, y2 = box
                    
                    # 발 위치 계산 (바운딩 박스 하단 중앙)
                    foot_x = (x1 + x2) / 2
                    foot_y = y2  # 바운딩 박스 하단 (발 위치)
                    
                    # 중심점도 계산 (표시용)
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    detections.append({
                        'id': int(track_id),
                        'bbox': [x1, y1, x2, y2],
                        'center': [center_x, center_y],  # 표시용 중심점
                        'foot': [foot_x, foot_y],        # 실제 위치 (발)
                        'confidence': conf,
                        'class': int(cls)
                    })
        
        return detections
    
    def convert_to_ground_coordinates(self, detections):
        """객체 좌표를 실제 지면 좌표로 변환 (발 좌표만)"""
        if self.homography_matrix is None:
            return detections
        
        ground_detections = []
        for det in detections:
            foot = det['foot']  # 발 위치만 사용
            
            # 발 위치를 2D 맵 좌표로 변환
            foot_array = np.array([[foot]], dtype=np.float32)
            map_foot = cv2.perspectiveTransform(foot_array, self.homography_matrix)
            map_foot = map_foot[0][0]
            
            # 픽셀 좌표를 실제 지면 좌표로 변환 (미터)
            real_x = map_foot[0] / self.pixels_per_meter
            real_y = map_foot[1] / self.pixels_per_meter
            
            # 발 위치만을 위한 작은 바운딩 박스 생성 (표시용)
            foot_size = 10  # 발 위치 표시를 위한 작은 크기
            map_x1 = map_foot[0] - foot_size
            map_y1 = map_foot[1] - foot_size
            map_x2 = map_foot[0] + foot_size
            map_y2 = map_foot[1] + foot_size
            
            ground_detections.append({
                'id': det['id'],
                'bbox': [map_x1, map_y1, map_x2, map_y2],  # 발 위치 중심의 작은 박스
                'real_position': [real_x, real_y],         # 실제 지면 좌표 (미터)
                'map_position': [map_foot[0], map_foot[1]], # 2D 맵 픽셀 좌표
                'confidence': det['confidence'],
                'class': det['class']
            })
        
        return ground_detections
    
    def create_2d_canvas_map(self):
        """실제 지면 크기를 기반으로 한 2D 캔버스 맵 생성"""
        if self.bev_size is None:
            width, height = 800, 600
        else:
            width, height = self.bev_size
        
        # 어두운 회색 배경의 캔버스 생성
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (40, 40, 40)  # 어두운 회색
        
        return canvas
    
    def draw_ground_grid(self, canvas):
        """실제 지면 거리 기준 격자 그리기"""
        height, width = canvas.shape[:2]
        
        # 격자 간격 (미터)
        grid_spacing_meters = 10  # 10미터 간격
        
        # 격자 간격을 픽셀로 변환
        grid_spacing_pixels = int(grid_spacing_meters * self.pixels_per_meter)
        
        # 수직 격자 (세로선)
        for i in range(0, width, grid_spacing_pixels):
            cv2.line(canvas, (i, 0), (i, height), (80, 80, 80), 1)
            # 실제 거리 표시
            distance_m = i / self.pixels_per_meter
            cv2.putText(canvas, f"{distance_m:.0f}m", (i+5, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        
        # 수평 격자 (가로선)
        for i in range(0, height, grid_spacing_pixels):
            cv2.line(canvas, (0, i), (width, i), (80, 80, 80), 1)
            # 실제 거리 표시
            distance_m = i / self.pixels_per_meter
            cv2.putText(canvas, f"{distance_m:.0f}m", (5, i-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        
        # 원점 표시
        cv2.circle(canvas, (0, 0), 5, (255, 255, 0), -1)  # 노란색
        cv2.putText(canvas, "O(0m,0m)", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 맵 정보 표시
        info_text = f"Map: {self.real_map_size['width_meters']:.1f}m x {self.real_map_size['height_meters']:.1f}m"
        cv2.putText(canvas, info_text, (width-300, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return canvas
    
    def draw_objects_on_canvas(self, canvas, detections):
        """2D 캔버스에 객체 정보 그리기 (발 좌표만 표시)"""
        for det in detections:
            obj_id = det['id']
            real_pos = det['real_position']
            map_pos = det['map_position']
            conf = det['confidence']
            cls = det['class']
            
            # 발 위치만 표시 (큰 원)
            pos_x, pos_y = map(int, map_pos)
            color = (0, 0, 255)  # 빨간색
            cv2.circle(canvas, (pos_x, pos_y), 8, color, -1)  # 발 위치 원
            
            # ID와 클래스 정보 표시 (발 위치 위에)
            label = f"ID:{obj_id} C:{cls}"
            cv2.putText(canvas, label, (pos_x-20, pos_y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 실제 좌표 정보 표시 (미터 단위, 발 위치 아래에)
            coord_text = f"({real_pos[0]:.1f}m, {real_pos[1]:.1f}m)"
            cv2.putText(canvas, coord_text, (pos_x-30, pos_y+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return canvas
    
    def draw_objects_on_original(self, frame, detections):
        """원본 프레임에 객체 정보 그리기"""
        for det in detections:
            obj_id = det['id']
            bbox = det['bbox']
            foot = det['foot']
            conf = det['confidence']
            cls = det['class']
            
            # 바운딩 박스 그리기
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 255, 0)  # 녹색
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # ID와 클래스 정보 표시
            label = f"ID:{obj_id} C:{cls}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 발 위치 표시 (큰 원, 빨간색)
            foot_x, foot_y = map(int, foot)
            cv2.circle(frame, (foot_x, foot_y), 5, (0, 0, 255), -1)  # 빨간색
            
            # 픽셀 좌표 정보 표시
            coord_text = f"({foot_x}, {foot_y})"
            cv2.putText(frame, coord_text, (x1, y2+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        return frame
    
    def run_dual_view(self):
        """원본 영상과 2D 맵을 별도 창으로 분리하여 표시"""
        if not os.path.exists(self.video_path):
            print(f"Video file not found: {self.video_path}")
            return False
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"Cannot open video file: {self.video_path}")
            return False
        
        # 비디오 정보
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Original video: {frame_width}x{frame_height}, {fps} FPS")
        print(f"2D map size: {self.bev_size[0]} x {self.bev_size[1]} pixels")
        print("Real-time monitoring mode started - Separate Windows")
        print("Controls:")
        print("  - 'q': Quit")
        print("  - 'p': Pause/Play")
        print("  - 's': Save screenshot")
        print("  - '1': Focus on Original Video window")
        print("  - '2': Focus on 2D Map window")
        
        paused = False
        frame_count = 0
        
        # 창 위치 설정 (화면을 분할하여 배치)
        cv2.namedWindow('Original Video', cv2.WINDOW_NORMAL)
        cv2.namedWindow('2D Ground Map', cv2.WINDOW_NORMAL)
        
        # 창 크기 및 위치 설정
        screen_width = 1920  # 일반적인 화면 너비
        screen_height = 1080  # 일반적인 화면 높이
        
        # 원본 영상 창 (왼쪽 절반)
        cv2.resizeWindow('Original Video', screen_width//2, screen_height//2)
        cv2.moveWindow('Original Video', 0, 0)
        
        # 2D 맵 창 (오른쪽 절반)
        cv2.resizeWindow('2D Ground Map', screen_width//2, screen_height//2)
        cv2.moveWindow('2D Ground Map', screen_width//2, 0)
        
        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("Video ended")
                    break
            
            if frame is None:
                continue
            
            frame_count += 1
            
            # 객체 감지 및 추적
            detections = self.detect_and_track_objects(frame)
            
            # 원본 프레임에 객체 그리기
            original_with_objects = self.draw_objects_on_original(frame.copy(), detections)
            
            # 2D 캔버스 맵 생성
            canvas_map = self.create_2d_canvas_map()
            
            # 실제 지면 좌표로 변환
            ground_detections = self.convert_to_ground_coordinates(detections)
            
            # 2D 캔버스에 격자 그리기
            canvas_map = self.draw_ground_grid(canvas_map)
            
            # 2D 캔버스에 객체 그리기
            canvas_with_objects = self.draw_objects_on_canvas(canvas_map.copy(), ground_detections)
            
            # 프레임 정보 추가
            cv2.putText(original_with_objects, f"Original Video - Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(original_with_objects, f"Objects: {len(detections)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(original_with_objects, f"Status: {'PAUSED' if paused else 'PLAYING'}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if paused else (0, 255, 0), 2)
            
            cv2.putText(canvas_with_objects, f"2D Ground Map - Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(canvas_with_objects, f"Objects: {len(detections)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(canvas_with_objects, f"Status: {'PAUSED' if paused else 'PLAYING'}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if paused else (0, 255, 0), 2)
            
            # 좌표 정보 콘솔 출력 (객체가 있을 때만)
            if ground_detections:
                print(f"\n=== Frame {frame_count} - Ground Coordinates (meters) ===")
                for det in ground_detections:
                    obj_id = det['id']
                    real_pos = det['real_position']
                    print(f"ID {obj_id}: ({real_pos[0]:.1f}m, {real_pos[1]:.1f}m)")
                print("=" * 50)
            
            # 별도 창에 표시
            cv2.imshow('Original Video', original_with_objects)
            cv2.imshow('2D Ground Map', canvas_with_objects)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Playing")
            elif key == ord('s'):
                # 스크린샷 저장 (두 창 모두)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                original_filename = f"original_video_{timestamp}.jpg"
                map_filename = f"2d_map_{timestamp}.jpg"
                cv2.imwrite(original_filename, original_with_objects)
                cv2.imwrite(map_filename, canvas_with_objects)
                print(f"Screenshots saved: {original_filename}, {map_filename}")
            elif key == ord('1'):
                # 원본 영상 창에 포커스
                cv2.setWindowProperty('Original Video', cv2.WND_PROP_TOPMOST, 1)
                cv2.setWindowProperty('2D Ground Map', cv2.WND_PROP_TOPMOST, 0)
                print("Focused on Original Video window")
            elif key == ord('2'):
                # 2D 맵 창에 포커스
                cv2.setWindowProperty('Original Video', cv2.WND_PROP_TOPMOST, 0)
                cv2.setWindowProperty('2D Ground Map', cv2.WND_PROP_TOPMOST, 1)
                print("Focused on 2D Map window")
        
        # 정리
        self.cap.release()
        cv2.destroyAllWindows()
        return True

def main():
    """메인 함수"""
    video_path = "smart-factory-peaceeye/app/test_video/final01.mp4"
    calibration_file = "final01_ground_truth_calibration.json"
    model_path = "smart-factory-peaceeye/app/models/weights/best_yolo11x.pt"
    # 2D 맵 시스템 생성 및 실행
    ground_map = GroundTruth2DMap(video_path, calibration_file, model_path)
    ground_map.run_dual_view()

if __name__ == "__main__":
    main()