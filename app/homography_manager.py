import cv2
import numpy as np
import json


class HomographyManager:
    """호모그래피 매트릭스 관리 클래스"""
    
    def __init__(self):
        self.homography_matrices = {}
        self.calibration_data = {}
        # Unity 맵 좌표계 설정 (4개 좌표 방식)
        self.unity_map_corners = {
            0: [  # 카메라 0 (복도) - 왼쪽 상단부터 시계방향
                {"x": 3700, "y": -4088},      # 왼쪽 상단
                {"x": 3700, "y": -1700},    # 오른쪽 상단
                {"x": 10700, "y": 1080},   # 오른쪽 하단
                {"x": 10700, "y": -8700}      # 왼쪽 하단
            ],
            1: [  # 카메라 1 (방) - 왼쪽 상단부터 시계방향
                {"x": 5439, "y": -770},    # 왼쪽 상단
                {"x": 4000, "y": -1350},    # 오른쪽 상단
                {"x": 3220, "y": 408},  # 오른쪽 하단
                {"x": 4606, "y": 903}   # 왼쪽 하단
            ]
        }
    
    def set_unity_map_corners(self, camera_id, corners):
        """Unity 맵에서 각 카메라 영역의 4개 모서리 좌표 설정
        corners: [왼쪽상단, 오른쪽상단, 오른쪽하단, 왼쪽하단] 순서
        """
        if len(corners) != 4:
            print(f"Error: Need exactly 4 corners for camera {camera_id}")
            return False
        
        self.unity_map_corners[camera_id] = corners
        print(f"Camera {camera_id} Unity corners set:")
        for i, corner in enumerate(corners):
            print(f"  Corner {i+1}: ({corner['x']}, {corner['y']})")
        return True
    
    def set_unity_map_corners_from_coords(self, camera_id, x1, y1, x2, y2, x3, y3, x4, y4):
        """Unity 맵에서 각 카메라 영역의 4개 모서리 좌표 설정 (개별 좌표 입력)"""
        corners = [
            {"x": x1, "y": y1},  # 왼쪽 상단
            {"x": x2, "y": y2},  # 오른쪽 상단
            {"x": x3, "y": y3},  # 오른쪽 하단
            {"x": x4, "y": y4}   # 왼쪽 하단
        ]
        return self.set_unity_map_corners(camera_id, corners)
    
    def add_camera_calibration(self, camera_id, calibration_file):
        """카메라별 캘리브레이션 파일 로드"""
        try:
            with open(calibration_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.homography_matrices[camera_id] = np.array(data['homography_matrix'])
            self.calibration_data[camera_id] = data
            
            print(f"Camera {camera_id} calibration loaded: {calibration_file}")
            return True
            
        except Exception as e:
            print(f"Failed to load calibration for camera {camera_id}: {e}")
            return False
    
    def transform_coordinates(self, camera_id, x, y):
        """좌표 변환 (호모그래피 + Unity 맵 4개 좌표 변환)"""
        if camera_id not in self.homography_matrices:
            print(f"Warning: No homography matrix for camera {camera_id}")
            return x, y
        
        try:
            # 1단계: 호모그래피 변환 (카메라 좌표 → 실제 지면 좌표)
            point = np.array([[x, y]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(point, self.homography_matrices[camera_id])
            real_x, real_y = transformed[0][0], transformed[0][1]
            
            # 2단계: Unity 맵 4개 좌표로 변환
            if camera_id in self.unity_map_corners:
                unity_corners = self.unity_map_corners[camera_id]
                
                # 실제 지면 좌표를 0~1 범위로 정규화 (캘리브레이션 데이터 기준)
                if camera_id == 0:  # 카메라 0 (final01)
                    # 7.5m × 7.8m 영역을 0~1로 정규화
                    norm_x = max(0, min(1, real_x / 7.5))
                    norm_y = max(0, min(1, real_y / 7.8))
                elif camera_id == 1:  # 카메라 1 (final02)
                    # 9.0m × 8.2m 영역을 0~1로 정규화
                    norm_x = max(0, min(1, real_x / 9.0))
                    norm_y = max(0, min(1, real_y / 8.2))
                else:
                    # 기본값 (0~1 범위로 가정)
                    norm_x = max(0, min(1, real_x))
                    norm_y = max(0, min(1, real_y))
                
                # 정규화된 좌표를 Unity 맵의 4개 모서리 좌표로 변환
                # 4개 모서리 좌표를 사용한 바이리니어 보간
                unity_x = (unity_corners[0]["x"] * (1 - norm_x) * (1 - norm_y) +  # 왼쪽 상단
                          unity_corners[1]["x"] * norm_x * (1 - norm_y) +         # 오른쪽 상단
                          unity_corners[2]["x"] * norm_x * norm_y +               # 오른쪽 하단
                          unity_corners[3]["x"] * (1 - norm_x) * norm_y)          # 왼쪽 하단
                
                unity_y = (unity_corners[0]["y"] * (1 - norm_x) * (1 - norm_y) +  # 왼쪽 상단
                          unity_corners[1]["y"] * norm_x * (1 - norm_y) +         # 오른쪽 상단
                          unity_corners[2]["y"] * norm_x * norm_y +               # 오른쪽 하단
                          unity_corners[3]["y"] * (1 - norm_x) * norm_y)          # 왼쪽 하단
                
                print(f"[UNITY] Camera {camera_id}: Image({x:.1f}, {y:.1f}) -> Real({real_x:.3f}, {real_y:.3f}) -> Norm({norm_x:.3f}, {norm_y:.3f}) -> Unity({unity_x:.1f}, {unity_y:.1f})")
                return unity_x, unity_y
            else:
                print(f"Warning: No Unity corners for camera {camera_id}")
                return real_x, real_y
                
        except Exception as e:
            print(f"Coordinate transformation failed: {e}")
            return x, y
    
    def get_calibration_info(self, camera_id):
        """캘리브레이션 정보 반환"""
        if camera_id in self.calibration_data:
            return self.calibration_data[camera_id]
        return None
    
    def get_unity_corners(self, camera_id):
        """Unity 맵 4개 모서리 좌표 반환"""
        return self.unity_map_corners.get(camera_id, None)
    
    def print_unity_corners(self, camera_id):
        """Unity 맵 4개 모서리 좌표 출력"""
        corners = self.get_unity_corners(camera_id)
        if corners:
            print(f"Camera {camera_id} Unity corners:")
            for i, corner in enumerate(corners):
                print(f"  Corner {i+1}: ({corner['x']}, {corner['y']})")
        else:
            print(f"No Unity corners set for camera {camera_id}")

