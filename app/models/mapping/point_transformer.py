import numpy as np

def transform_point(pointx,pointz, homography_matrix):
    """
    제공된 호모그래피 행렬을 사용하여 이미지 좌표의 2D 점을
    실제 세계 좌표로 변환합니다.

    Args:
        point (tuple or list): 이미지의 점 (x, y) 좌표입니다.
        homography_matrix (numpy.ndarray): 3x3 호모그래피 행렬입니다.
        [[1,2,3], [1,2,3], [1,2,3]]

    Returns:
        tuple: 변환된 실제 세계의 (x, y) 좌표입니다.
               변환이 불가능한 경우 (0, 0)을 반환합니다.
    """
    try:
        # 점을 동차 좌표(x, y, 1)로 변환합니다.
        point_homogeneous = np.array([pointx, pointz, 1], dtype="float32")

        # 행렬 곱셈을 사용하여 호모그래피 변환을 적용합니다.
        transformed_point_homogeneous = homography_matrix @ point_homogeneous

        # 마지막 요소(w)로 나누어 동차 좌표에서 데카르트 좌표로 다시 변환(정규화)합니다.
        w = transformed_point_homogeneous[2]
        if w == 0:
            # 0으로 나누는 것을 방지합니다.
            return (0, 0)

        transformed_x = transformed_point_homogeneous[0] / w
        transformed_y = transformed_point_homogeneous[1] / w

        return transformed_x, transformed_y
    except Exception as e:
        # 변환 중 오류 발생시 (0, 0) 반환
        print(f"Coordinate transformation error: {e}")
        return (0, 0)

"""
--- 사용 예시 ---
if __name__ == '__main__':
    # 예시를 위해 요청된 대로 3x3 영행렬을 생성합니다.
    # 실제 애플리케이션에서는 계산된 호모그래피 행렬을 전달해야 합니다.
    print("--- 3x3 영행렬을 사용한 예시 ---")
    H_matrix_zero = np.zeros((3, 3))
    print(H_matrix_zero)

    # 이미지의 예시 점
    image_point = (150, 250)

    # 함수를 사용하여 점을 변환합니다.
    real_world_coord = transform_point(image_point, H_matrix_zero)

    if real_world_coord is not None:
        print(f"\n이미지 좌표 {image_point} -> 변환된 좌표: {real_world_coord}")
    else:
        print(f"\n좌표 {image_point}를 변환할 수 없습니다.")

    # 보다 현실적인 변환을 보여주기 위해 0이 아닌 행렬을 사용한 예시
    print("\n--- 0이 아닌 행렬을 사용한 예시 ---")
    H_matrix_example = np.array([
        [-2.55448371e-01, -1.03075466e+00,  4.09533540e+02],
        [ 1.03907352e-01, -1.54835632e-01,  2.29977585e+02],
        [ 2.62937216e-04, -2.03178889e-03,  1.00000000e+00]
    ])
    print(H_matrix_example)
    real_world_coord_2 = transform_point(image_point, H_matrix_example)
    if real_world_coord_2 is not None:
        print(f"\n이미지 좌표 {image_point} -> 변환된 좌표: {real_world_coord_2}")
    else:
        print(f"\n좌표 {image_point}를 변환할 수 없습니다.")
"""