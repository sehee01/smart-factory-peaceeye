import cv2
import numpy as np
import json

class HomographyCalibrator:
    STATE_COLLECTING_POINTS = 1
    STATE_CALIBRATED = 3

    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.pixel_points = []
        self.real_points = []
        self.homography_matrix = None
        self.state = self.STATE_COLLECTING_POINTS
        self.error_message = ""

    def add_point(self, pixel_point):
        """픽셀 좌표를 추가하고, 터미널에서 실제 좌표 입력을 요청"""
        if self.state != self.STATE_COLLECTING_POINTS:
            return

        self.pixel_points.append(pixel_point)
        print(f"\n선택된 픽셀 좌표: {pixel_point}")
        
        while True:
            try:
                input_str = input(f"  => P{len(self.pixel_points)}의 실제 좌표 (X, Y)를 콤마로 구분하여 입력하세요: ")
                coords = [float(val.strip()) for val in input_str.split(',')]
                if len(coords) != 2:
                    raise ValueError("X, Y 두 개의 값을 입력해야 합니다.")
                
                self.real_points.append(coords)
                self.error_message = ""
                break  # 올바른 입력을 받으면 루프 종료
            except ValueError as e:
                print(f"    [입력 오류] {e}. 다시 입력해주세요.")

    def calculate_homography(self):
        if len(self.pixel_points) < 4:
            self.error_message = "오류: 최소 4개의 대응점이 필요합니다."
            print(self.error_message)
            return False
            
        src_points = np.array(self.pixel_points, dtype=np.float32)
        dst_points = np.array(self.real_points, dtype=np.float32)
        
        matrix, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        
        if matrix is not None:
            self.homography_matrix = matrix
            self.state = self.STATE_CALIBRATED
            self.error_message = ""
            print("\nHomography 행렬이 성공적으로 계산되었습니다.")
            self.save_calibration()
            return True
        else:
            self.error_message = "오류: Homography 계산 실패. 점 배치를 확인하세요."
            print(self.error_message)
            return False

    def draw_ui(self, frame):
        display_frame = frame.copy()
        
        if self.error_message:
            cv2.putText(display_frame, self.error_message, (20, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for i, point in enumerate(self.pixel_points):
            cv2.circle(display_frame, tuple(point), 5, (0, 255, 0), -1)
            cv2.putText(display_frame, f"P{i+1}", (point[0] + 10, point[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if self.state == self.STATE_COLLECTING_POINTS:
            cv2.putText(display_frame, f"Points: {len(self.pixel_points)}. Click to add more.", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        elif self.state == self.STATE_CALIBRATED:
            cv2.putText(display_frame, "Calibration Complete! Press 's' to save, 'q' to quit", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return display_frame

    def save_calibration(self):
        if self.homography_matrix is None:
            print("저장할 캘리브레이션 데이터가 없습니다.")
            return

        data = {
            'camera_id': self.camera_id,
            'pixel_points': self.pixel_points,
            'real_points': self.real_points,
            'homography_matrix': self.homography_matrix.tolist()
        }
        filename = f'camera_{self.camera_id}_calibration.json'
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"캘리브레이션 데이터가 '{filename}' 파일로 저장되었습니다.")

    def load_calibration(self):
        filename = f'camera_{self.camera_id}_calibration.json'
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.pixel_points = data['pixel_points']
            self.real_points = data['real_points']
            self.homography_matrix = np.array(data['homography_matrix'])
            self.state = self.STATE_CALIBRATED
            print(f"'{filename}'에서 캘리브레이션 데이터를 불러왔습니다.")
            return True
        except FileNotFoundError:
            print(f"캘리브레이션 파일('{filename}')을 찾을 수 없습니다.")
            return False

    def reset_points(self):
        self.pixel_points = []
        self.real_points = []
        self.homography_matrix = None
        self.state = self.STATE_COLLECTING_POINTS
        self.error_message = ""
        print("\n모든 대응점이 초기화되었습니다. 다시 시작하세요.")