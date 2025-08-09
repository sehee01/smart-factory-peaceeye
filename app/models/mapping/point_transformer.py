import numpy as np
import cv2

def transform_point(x, y, homography_matrix):
    """
    이미지 좌표를 실제 좌표로 변환하는 함수
    
    Args:
        x, y: 이미지 좌표
        homography_matrix: 호모그래피 변환 행렬 (3x3)
    
    Returns:
        real_x, real_y: 실제 좌표
    """
    # 입력 좌표를 동차 좌표로 변환
    point = np.array([[x], [y], [1]], dtype=np.float64)
    
    # 호모그래피 변환 적용
    transformed = np.dot(homography_matrix, point)
    
    # 동차 좌표를 일반 좌표로 변환
    real_x = transformed[0, 0] / transformed[2, 0]
    real_y = transformed[1, 0] / transformed[2, 0]
    
    return real_x, real_y
