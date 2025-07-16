import cv2
from datetime import datetime
from homography_calibration import HomographyCalibrator

def save_coordinates_to_txt(calibrator, filename=None):
    """수집된 좌표와 Homography 행렬을 텍스트 파일로 상세히 저장"""
    if not calibrator.pixel_points:
        print("저장할 좌표 데이터가 없습니다.")
        return

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"calibration_coords_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=== Homography Calibration Report ===\n")
        f.write(f"Camera ID: {calibrator.camera_id}\n")
        f.write(f"Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Correspondence Points: {len(calibrator.pixel_points)}\n\n")
        
        f.write("Pixel -> Real World Coordinate Mapping:\n")
        f.write("-" * 50 + "\n")
        for i, (p_pt, r_pt) in enumerate(zip(calibrator.pixel_points, calibrator.real_points)):
            f.write(f"  Point {i+1}: Pixel({p_pt[0]}, {p_pt[1]}) -> Real({r_pt[0]}, {r_pt[1]})\n")
        
        f.write("\nHomography Matrix:\n")
        f.write("-" * 30 + "\n")
        if calibrator.homography_matrix is not None:
            for row in calibrator.homography_matrix:
                f.write(f"  [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}]\n")
        else:
            f.write("  Matrix not calculated.\n")

    print(f"좌표 데이터가 '{filename}' 파일로 저장되었습니다.")


def run_calibration_process():
    video_path = "test_vedio/0_te3.mp4" #영상주소
    camera_id = 0                       #카메라 id
    window_name = f'Homography Calibration (Camera {camera_id})'

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: 영상 파일({video_path})을 열 수 없습니다.")
        return

    calibrator = HomographyCalibrator(camera_id)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            calibrator.add_point([x, y])

    if calibrator.load_calibration():
        print("\n기존 캘리브레이션 데이터를 불러왔습니다.")
        use_existing = input("기존 데이터를 사용하시겠습니까? (y/n): ").lower()
        if use_existing != 'y':
            calibrator.reset_points()

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n=== 캘리브레이션 시작 ===")
    print(" 조작 키:")
    print("  - 마우스 클릭: 대응점 선택 (터미널에 좌표 입력)")
    print("  - 'c': Homography 계산 (4점 이상 선택 시)")
    print("  - 'r': 모든 점 초기화")
    print("  - 's': 현재 정보를 TXT 파일로 저장")
    print("  - 'q': 종료")
    print("=========================")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        display_frame = calibrator.draw_ui(frame)
        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            if calibrator.calculate_homography():
                save_coordinates_to_txt(calibrator)
        elif key == ord('r'):
            calibrator.reset_points()
        elif key == ord('s'):
            save_coordinates_to_txt(calibrator)

    cap.release()
    cv2.destroyAllWindows()
    print("프로그램을 종료합니다.")

if __name__ == "__main__":
    run_calibration_process()
