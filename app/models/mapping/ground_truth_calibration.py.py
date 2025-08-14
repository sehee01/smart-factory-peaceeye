import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Optional

class GroundTruthCalibration:
    """
    실제 지면 길이를 기반으로 한 homography 캘리브레이션 도구
    영상에서 4개 점을 클릭하고 실제 지면 거리를 입력하여 정확한 스케일을 설정
    """
    
    def __init__(self, video_path: str, output_file: str = "ground_truth_calibration.json"):
        self.video_path = video_path
        self.output_file = output_file
        self.cap = None
        self.frame = None
        self.points = []  # 클릭한 4개 점 (픽셀 좌표)
        self.real_distances = []  # 실제 지면 거리 (미터)
        self.current_point = 0
        self.homography_matrix = None
        self.bev_size = None
        
        # UI 상태
        self.drawing = False
        self.show_instructions = True
        
    def mouse_callback(self, event, x, y, flags, param):
        """마우스 클릭 이벤트 처리"""
        if event == cv2.EVENT_LBUTTONDOWN and self.current_point < 4:
            # 점 추가
            self.points.append([x, y])
            print(f"Point {self.current_point + 1}: ({x}, {y})")
            self.current_point += 1
            
            # 4개 점이 모두 선택되면 실제 거리 입력 요청
            if self.current_point == 4:
                self.get_real_distances()
    
    def get_real_distances(self):
        """사용자로부터 실제 지면 거리 입력받기"""
        print("\n=== Ground Truth Distance Input ===")
        print("Enter the actual ground distances in meters between 4 points.")
        print("Point order: Top-Left → Top-Right → Bottom-Right → Bottom-Left")
        
        self.real_distances = []
        
        # 각 변의 실제 거리 입력
        distances = [
            "Top-Left to Top-Right (width distance, meters): ",
            "Top-Right to Bottom-Right (height distance, meters): ",
            "Bottom-Right to Bottom-Left (width distance, meters): ",
            "Bottom-Left to Top-Left (height distance, meters): "
        ]
        
        for i, prompt in enumerate(distances):
            while True:
                try:
                    distance = float(input(prompt))
                    if distance > 0:
                        self.real_distances.append(distance)
                        break
                    else:
                        print("Distance must be greater than 0.")
                except ValueError:
                    print("Please enter a valid number.")
        
        # BEV 맵 크기 결정
        self.determine_bev_size()
        
        # Homography 계산
        self.calculate_homography()
        
        # 결과 저장
        self.save_calibration()
        
        print("\nCalibration completed!")
        print(f"Results saved to '{self.output_file}'.")
    
    def determine_bev_size(self):
        """실제 거리를 기반으로 BEV 맵 크기 결정"""
        # 실제 거리의 최대값을 기준으로 픽셀 스케일 결정
        max_real_distance = max(self.real_distances)
        
        # 1미터당 픽셀 수 결정 (화면 크기에 맞게 조정)
        pixels_per_meter = 50  # 1미터당 50픽셀
        
        # 실제 맵의 크기 계산
        real_width = max(self.real_distances[0], self.real_distances[2])  # 가로 거리 중 큰 값
        real_height = max(self.real_distances[1], self.real_distances[3])  # 세로 거리 중 큰 값
        
        # BEV 맵 크기 계산
        bev_width = int(real_width * pixels_per_meter)
        bev_height = int(real_height * pixels_per_meter)
        
        self.bev_size = [bev_width, bev_height]
        
        print(f"\nReal map size: {real_width:.1f}m x {real_height:.1f}m")
        print(f"BEV map size: {bev_width} x {bev_height} pixels")
        print(f"Scale: 1 meter = {pixels_per_meter} pixels")
    
    def calculate_homography(self):
        """Homography 행렬 계산"""
        if len(self.points) != 4:
            print("4개 점이 필요합니다.")
            return False
        
        # 실제 맵에서의 목표 좌표 계산
        real_width = max(self.real_distances[0], self.real_distances[2])
        real_height = max(self.real_distances[1], self.real_distances[3])
        
        # BEV 맵에서의 목표 좌표 (픽셀)
        pixels_per_meter = 50
        target_width = int(real_width * pixels_per_meter)
        target_height = int(real_height * pixels_per_meter)
        
        # 목표 좌표 (왼쪽상단 → 오른쪽상단 → 오른쪽하단 → 왼쪽하단)
        target_points = np.array([
            [0, 0],                    # 왼쪽상단
            [target_width, 0],         # 오른쪽상단
            [target_width, target_height],  # 오른쪽하단
            [0, target_height]         # 왼쪽하단
        ], dtype=np.float32)
        
        # 원본 좌표
        source_points = np.array(self.points, dtype=np.float32)
        
        # Homography 행렬 계산
        self.homography_matrix = cv2.getPerspectiveTransform(source_points, target_points)
        
        print(f"\nHomography matrix calculation completed")
        print(f"Source points: {self.points}")
        print(f"Target points: {target_points.tolist()}")
        
        return True
    
    def save_calibration(self):
        """캘리브레이션 결과 저장"""
        if self.homography_matrix is None:
            print("Homography matrix has not been calculated.")
            return False
        
        # 실제 맵 좌표 계산
        real_width = max(self.real_distances[0], self.real_distances[2])
        real_height = max(self.real_distances[1], self.real_distances[3])
        
        real_points = [
            [0, 0],                    # 왼쪽상단
            [real_width, 0],           # 오른쪽상단
            [real_width, real_height], # 오른쪽하단
            [0, real_height]           # 왼쪽하단
        ]
        
        calibration_data = {
            "points": self.points,
            "real_points": real_points,
            "real_distances": self.real_distances,
            "homography_matrix": self.homography_matrix.tolist(),
            "bev_size": self.bev_size,
            "pixels_per_meter": 50,
            "real_map_size": {
                "width_meters": real_width,
                "height_meters": real_height
            },
            "calibration_info": {
                "method": "ground_truth_calibration",
                "description": "Homography calibration based on actual ground distances"
            }
        }
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, indent=2, ensure_ascii=False)
        
        return True
    
    def draw_ui(self, frame):
        """UI 그리기"""
        display_frame = frame.copy()
        
        # 지시사항 표시
        if self.show_instructions:
            instructions = [
                "Ground Truth Calibration",
                "",
                "Click 4 ground corners in order:",
                "1. Top-Left",
                "2. Top-Right", 
                "3. Bottom-Right",
                "4. Bottom-Left",
                "",
                f"Selected points: {self.current_point}/4",
                "",
                "ESC: Exit"
            ]
            
            y_offset = 30
            for i, text in enumerate(instructions):
                cv2.putText(display_frame, text, (10, y_offset + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, text, (10, y_offset + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # 선택된 점들 그리기
        for i, point in enumerate(self.points):
            x, y = point
            color = (0, 255, 0) if i < self.current_point else (0, 0, 255)
            cv2.circle(display_frame, (x, y), 8, color, -1)
            cv2.circle(display_frame, (x, y), 10, (255, 255, 255), 2)
            cv2.putText(display_frame, str(i + 1), (x + 15, y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 점들을 선으로 연결
        if len(self.points) >= 2:
            for i in range(len(self.points) - 1):
                cv2.line(display_frame, tuple(self.points[i]), tuple(self.points[i + 1]), 
                        (255, 255, 0), 2)
            
            # 마지막 점과 첫 번째 점 연결
            if len(self.points) == 4:
                cv2.line(display_frame, tuple(self.points[3]), tuple(self.points[0]), 
                        (255, 255, 0), 2)
        
        return display_frame
    
    def run(self):
        """캘리브레이션 실행"""
        if not os.path.exists(self.video_path):
            print(f"비디오 파일을 찾을 수 없습니다: {self.video_path}")
            return False
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"비디오 파일을 열 수 없습니다: {self.video_path}")
            return False
        
        # 첫 번째 프레임 읽기
        ret, self.frame = self.cap.read()
        if not ret:
            print("비디오에서 프레임을 읽을 수 없습니다.")
            return False
        
        # 윈도우 생성 및 마우스 콜백 설정
        window_name = "Ground Truth Calibration"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("Ground Truth Calibration started")
        print("Click 4 ground corners in order.")
        print("Top-Left → Top-Right → Bottom-Right → Bottom-Left")
        
        while True:
            # UI 그리기
            display_frame = self.draw_ui(self.frame)
            
            # 화면에 표시
            cv2.imshow(window_name, display_frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('r'):  # R키로 재시작
                self.points = []
                self.current_point = 0
                self.real_distances = []
                self.homography_matrix = None
                print("Calibration restarted")
            elif key == ord('h'):  # H키로 지시사항 토글
                self.show_instructions = not self.show_instructions
            
            # 4개 점이 모두 선택되면 종료
            if self.current_point >= 4:
                break
        
        # 정리
        self.cap.release()
        cv2.destroyAllWindows()
        
        return self.homography_matrix is not None

def main():
    """메인 함수"""
    video_path = "smart-factory-peaceeye/app/test_video/final02.mp4"
    output_file = f"{video_path.split('/')[-1].split('.')[0]}_ground_truth_calibration.json"
    
    # 캘리브레이션 실행
    calibrator = GroundTruthCalibration(video_path, output_file)
    success = calibrator.run()
    
    if success:
        print(f"\nCalibration successful!")
        print(f"Result file: {output_file}")
    else:
        print("\nCalibration failed")

if __name__ == "__main__":
    main()