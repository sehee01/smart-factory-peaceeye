#!/usr/bin/env python3
"""
제한구역 설정 도구 간단 테스트
"""

import os
import sys

def test_video_path():
    """비디오 파일 경로 테스트"""
    print("=== 비디오 파일 경로 테스트 ===")
    
    # main2.py와 동일한 기본 경로
    default_video = "../test_video/KSEB03.mp4"
    abs_path = os.path.abspath(default_video)
    
    print(f"기본 비디오 경로: {default_video}")
    print(f"절대 경로: {abs_path}")
    print(f"파일 존재: {'✅' if os.path.exists(default_video) else '❌'}")
    
    if not os.path.exists(default_video):
        print("\n[WARNING] 기본 비디오 파일을 찾을 수 없습니다.")
        print("사용 가능한 비디오 파일을 확인하세요:")
        os.system("python zone_coordinate_picker.py --list-videos")
        return False
    
    return True

def test_zone_picker():
    """제한구역 설정 도구 테스트"""
    print("\n=== 제한구역 설정 도구 테스트 ===")
    
    if not test_video_path():
        return False
    
    print("\n[INFO] 제한구역 설정 도구를 실행합니다...")
    print("사용법:")
    print("1. 마우스로 드래그하여 제한구역을 그리세요")
    print("2. S 키를 눌러 JSON 파일로 저장하세요")
    print("3. Q 키를 눌러 종료하세요")
    
    try:
        # zone_coordinate_picker.py 실행
        os.system("python zone_coordinate_picker.py")
        return True
    except KeyboardInterrupt:
        print("\n[INFO] 사용자에 의해 중단되었습니다.")
        return False
    except Exception as e:
        print(f"[ERROR] 도구 실행 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("제한구역 설정 도구 간단 테스트를 시작합니다...\n")
    
    # 현재 디렉토리 확인
    print(f"[INFO] 현재 작업 디렉토리: {os.getcwd()}")
    
    # 테스트 실행
    success = test_zone_picker()
    
    if success:
        print("\n🎉 테스트가 완료되었습니다!")
        print("이제 main2.py에서 제한구역 설정을 사용할 수 있습니다.")
    else:
        print("\n⚠️ 테스트가 실패했습니다.")
        print("문제를 해결한 후 다시 시도하세요.")

if __name__ == "__main__":
    main()
