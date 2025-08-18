#!/usr/bin/env python3
"""
Ultralytics Tracking 테스트 스크립트
ByteTrack 대신 Ultralytics의 내장 tracking 기능을 테스트합니다.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRA_PATHS = [
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "deep-person-reid-master"),
    str(PROJECT_ROOT / "app" / "models" / "mapping"),
]

for p in EXTRA_PATHS:
    if p not in sys.path:
        sys.path.insert(0, p)

from detector.ultralytics_tracker import UltralyticsTrackerManager
from config import settings


def test_ultralytics_tracking():
    """Ultralytics tracking 기능 테스트"""
    print("🧪 Testing Ultralytics Tracking...")
    
    # 테스트 설정
    model_path = "models/weights/video.pt"
    tracker_config = {
        "target_width": 640,
        "track_buffer": 30,
        "frame_rate": 30
    }
    
    # Ultralytics Tracker 초기화
    tracker = UltralyticsTrackerManager(model_path, tracker_config)
    print(f"✅ Tracker initialized with model: {model_path}")
    
    # 테스트 비디오 파일
    test_video = "test_video/TEST100.mp4"
    
    try:
        cap = cv2.VideoCapture(test_video)
        if not cap.isOpened():
            print(f"❌ Could not open video: {test_video}")
            return
        
        print(f"✅ Opened test video: {test_video}")
        
        frame_count = 0
        max_frames = 3000  # 테스트용으로 100프레임만 처리
        
        import time
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 처리 시간 측정 시작
            start_time = time.time()
            
            # Ultralytics tracking 실행
            track_list = tracker.detect_and_track(frame, frame_count)
            
            # 처리 시간 측정 끝
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000  # ms로 변환
            fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
            
            # 결과 시각화
            for track in track_list:
                track_id = track["track_id"]
                bbox = track["bbox"]
                confidence = track.get("confidence", 0.0)
                
                x1, y1, x2, y2 = map(int, bbox)
                
                # 바운딩 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID:{track_id} ({confidence:.2f})', 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 프레임 정보 표시
            cv2.putText(frame, f'Frame: {frame_count}, Tracks: {len(track_list)}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f'Time: {processing_time:.1f}ms, FPS: {fps:.1f}', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 결과 출력 (첫 10프레임만)
            if frame_count <= 10:
                print(f"Frame {frame_count}: {len(track_list)} tracks detected - Time: {processing_time:.1f}ms, FPS: {fps:.1f}")
                for track in track_list:
                    print(f"  - Track ID: {track['track_id']}, Confidence: {track.get('confidence', 0.0):.2f}")
            
            # 모든 프레임의 처리 정보 출력 (간격 조정)
            if frame_count % 30 == 0:  # 30프레임마다 출력
                print(f"Frame {frame_count}: {len(track_list)} objects - Time: {processing_time:.1f}ms, FPS: {fps:.1f}")
            
            # 화면에 표시
            cv2.imshow('Ultralytics Tracking Test', frame)
            
            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"✅ Test completed! Processed {frame_count} frames")
        print(f"📊 Final track history count: {len(tracker.track_history)}")
        
        # 성능 통계 출력
        if frame_count > 0:
            avg_fps = frame_count / (time.time() - start_time) if 'start_time' in locals() else 0
            print(f"📈 Average FPS: {avg_fps:.1f}")
            print(f"📊 Total unique track IDs: {len(tracker.track_history)}")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


def test_track_history():
    """Track history 기능 테스트"""
    print("\n🧪 Testing Track History...")
    
    model_path = "models/weights/bestcctv.pt"
    tracker_config = {"target_width": 640, "track_buffer": 5}
    
    tracker = UltralyticsTrackerManager(model_path, tracker_config)
    
    # 가상의 track history 추가
    test_track_id = 1
    for i in range(10):
        bbox = [100 + i*10, 100 + i*5, 150 + i*10, 200 + i*5]
        tracker._update_track_history(test_track_id, bbox, i)
    
    # history 조회
    history = tracker.get_track_history(test_track_id)
    print(f"✅ Track {test_track_id} history length: {len(history)}")
    print(f"   Expected max length: {tracker.max_history_length}")
    
    # history 내용 확인
    for i, entry in enumerate(history):
        print(f"   Entry {i}: Frame {entry['frame_id']}, Center {entry['center']}")


if __name__ == "__main__":
    print("🚀 Starting Ultralytics Tracking Tests...")
    print("=" * 50)
    
    # 기본 tracking 테스트
    test_ultralytics_tracking()
    
    # Track history 테스트
    test_track_history()
    
    print("\n🎉 All tests completed!")
    print("💡 To run the full system, use: python new_main_ultra.py")
