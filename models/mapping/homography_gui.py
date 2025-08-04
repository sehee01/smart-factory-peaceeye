import cv2
import os
from datetime import datetime
from homography_calibration2 import HomographyCalibrator
import argparse

def save_coordinates_to_txt(calibrator, filename=None):
    if not calibrator.pixel_points or not calibrator.real_points:
        print("저장할 좌표 데이터가 없습니다.")
        return

    if filename is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "..", "result", "matrix")
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"homography_matrix_{calibrator.camera_id}.txt")

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=== Homography Calibration Report ===\n")
        f.write(f"Camera ID: {calibrator.camera_id}\n")
        f.write(f"Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Correspondence Points: {len(calibrator.pixel_points)}\n\n")
        f.write("Pixel -> Real World Coordinate Mapping:\n")
        f.write("-" * 50 + "\n")
        for i, (pix, real) in enumerate(zip(calibrator.pixel_points, calibrator.real_points)):
            f.write(f"  Point {i+1}: Pixel({pix[0]}, {pix[1]}) -> Real({real[0]}, {real[1]})\n")
        f.write("\nHomography Matrix:\n")
        f.write("-" * 30 + "\n")
        if calibrator.homography_matrix is not None:
            for row in calibrator.homography_matrix:
                f.write(f"  [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}]\n")
        else:
            f.write("Matrix not calculated.\n")

    print(f"좌표 데이터가 '{filename}' 파일로 저장되었습니다.")

def run_calibration_process(video_path, camera_id):
    window_name = f'Homography Calibration (Camera {camera_id})'

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: 영상 파일({video_path})을 열 수 없습니다.")
        return

    calibrator = HomographyCalibrator(camera_id=camera_id)

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
    print("  - 마우스 클릭: 대응점 선택")
    print("  - 'c': Homography 계산 및 실제 좌표 입력")
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
            calibrator.input_real_points()
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
    parser = argparse.ArgumentParser(description='Homography Calibration GUI')
    parser.add_argument('--video', type=str, required=True, help='Path to the video file')
    parser.add_argument('--camera_id', type=int, required=True, help='ID of the camera')
    args = parser.parse_args()
    run_calibration_process(args.video, args.camera_id)
