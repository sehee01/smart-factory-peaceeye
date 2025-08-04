import numpy as np
import cv2
import json

class HomographyCalibrator:
    STATE_COLLECTING_POINTS = 1
    STATE_CALIBRATED = 3

    def __init__(self, camera_id=None, homography_matrix=None):
        self.camera_id = camera_id
        self.pixel_points = []
        self.real_points = []
        self.homography_matrix = homography_matrix
        self.state = self.STATE_COLLECTING_POINTS if homography_matrix is None else self.STATE_CALIBRATED
        self.error_message = ""

    def set_homography_matrix(self, matrix):
        self.homography_matrix = matrix
        self.state = self.STATE_CALIBRATED

    def pixel_to_map_coordinates(self, pixel_point):
        if self.homography_matrix is None:
            raise ValueError("Homography matrix has not been set.")
        x, y = pixel_point
        pixel_homog = np.array([x, y, 1.0])
        map_point = self.homography_matrix @ pixel_homog
        map_point /= map_point[2]  # homogeneous normalization
        return map_point[0], map_point[1]

    def add_point(self, pixel_point):
        if self.state != self.STATE_COLLECTING_POINTS:
            return
        self.pixel_points.append(pixel_point)
        print(f"\n선택된 픽셀 좌표: {pixel_point}")

    def input_real_points(self):
        if len(self.pixel_points) == 0:
            print("입력된 픽셀 좌표가 없습니다.")
            return
        self.real_points = []
        for i, p in enumerate(self.pixel_points):
            while True:
                try:
                    input_str = input(f"  => P{i+1} ({p[0]}, {p[1]})의 실제 좌표 (X, Y)를 입력하세요: ")
                    coords = [float(val.strip()) for val in input_str.split(',')]
                    if len(coords) != 2:
                        raise ValueError("X, Y 두 개의 값을 입력해야 합니다.")
                    self.real_points.append(coords)
                    break
                except ValueError as e:
                    print(f"    [입력 오류] {e}. 다시 입력해주세요.")

    def calculate_homography(self):
        if len(self.pixel_points) < 4 or len(self.real_points) < 4:
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
            cv2.putText(display_frame, self.error_message, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for i, point in enumerate(self.pixel_points):
            cv2.circle(display_frame, tuple(point), 5, (0, 255, 0), -1)
            cv2.putText(display_frame, f"P{i+1}", (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        status_text = (
            f"Points: {len(self.pixel_points)}. Click to add more."
            if self.state == self.STATE_COLLECTING_POINTS
            else "Calibration Complete! Press 's' to save, 'q' to quit"
        )
        color = (255, 255, 255) if self.state == self.STATE_COLLECTING_POINTS else (0, 255, 255)
        cv2.putText(display_frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

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
