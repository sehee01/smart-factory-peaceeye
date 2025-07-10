import os
import cv2
from ultralytics import YOLO
import locale

# 터미널 환경에서 인코딩 문제 방지 (Windows용)
locale.getpreferredencoding = lambda: "UTF-8"

# 현재 .py 파일 기준으로 절대 경로 생성
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, '..', 'weights', 'yolo.pt')
model = YOLO(model_path)

# 웹캠 켜기
cap = cv2.VideoCapture(0)

# 웹캠 오류 처리
if not cap.isOpened():
    print("Camera open failed")
    exit()

# 비디오 저장 설정
output_filename = '../result/video/output_result2.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 20.0
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

exit_flag = False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # YOLO 추론
        results = model(frame)

        # 결과 프레임 (Bounding Box 포함)
        annotated_frame = results[0].plot()

        # 결과 화면 출력
        cv2.imshow('YOLO Detection', annotated_frame)

        # 결과 저장
        out.write(annotated_frame)

        # 박스 정보 출력 (필요 시 사용)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = box.cls[0].item()
                # print(f"Class: {cls}, Confidence: {conf:.2f}, Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

                # 종료 키 체크
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    exit_flag = True
                    break
            if exit_flag:
                break

        if exit_flag:
            break

finally:
    # 자원 해제
    cap.release()
    out.release()
    cv2.destroyAllWindows()
