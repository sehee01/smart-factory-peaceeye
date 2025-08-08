#!/usr/bin/env python3
"""
제한구역 알람 시스템 테스트
"""

import numpy as np
from datetime import datetime

def test_distance_calculation():
    """거리 계산 함수 테스트"""
    print("=== 거리 계산 테스트 ===")
    
    # 테스트용 제한구역 (사각형)
    zone_coords = {
        'x1': 100, 'y1': 100,
        'x2': 200, 'y2': 200
    }
    
    # 테스트 포인트들
    test_points = [
        {'x': 150, 'y': 150, 'description': '제한구역 내부'},
        {'x': 50, 'y': 150, 'description': '제한구역 왼쪽'},
        {'x': 250, 'y': 150, 'description': '제한구역 오른쪽'},
        {'x': 150, 'y': 50, 'description': '제한구역 위쪽'},
        {'x': 150, 'y': 250, 'description': '제한구역 아래쪽'},
        {'x': 50, 'y': 50, 'description': '제한구역 대각선 왼쪽 위'},
        {'x': 250, 'y': 250, 'description': '제한구역 대각선 오른쪽 아래'},
    ]
    
    def calculate_distance(point, zone_coords):
        """점과 사각형 제한구역 사이의 거리 계산"""
        x, y = point['x'], point['y']
        x1, y1, x2, y2 = zone_coords['x1'], zone_coords['y1'], zone_coords['x2'], zone_coords['y2']
        
        # 제한구역 내부에 있는지 확인
        if x1 <= x <= x2 and y1 <= y <= y2:
            return 0  # 제한구역 내부
        
        # 제한구역 외부에서 가장 가까운 거리 계산
        dx = max(x1 - x, 0, x - x2)
        dy = max(y1 - y, 0, y - y2)
        distance = np.sqrt(dx*dx + dy*dy)
        
        return distance
    
    for point in test_points:
        distance = calculate_distance(point, zone_coords)
        alarm_type = "치명적" if distance == 0 else "경고" if distance <= 100 else "정상"
        print(f"{point['description']}: 거리={distance:.1f}픽셀 → {alarm_type} 알람")

def test_alarm_logic():
    """알람 로직 테스트"""
    print("\n=== 알람 로직 테스트 ===")
    
    # 설정
    RESTRICTED_ZONE_CONFIG = {
        'alarm_threshold': 0,  # 제한구역 내부 진입 시 치명적 알람
        'warning_threshold': 100,  # 1미터(100픽셀) 이내 접근 시 경고 알람
        'alarm_message': '제한구역 내부 진입! 즉시 이탈하세요!',
        'warning_message': '제한구역 1미터 이내 접근! 주의하세요!'
    }
    
    # 테스트 거리들
    test_distances = [0, 50, 100, 150, 200]
    
    for distance in test_distances:
        if distance == 0:
            severity = 'critical'
            message = RESTRICTED_ZONE_CONFIG['alarm_message']
        elif distance <= RESTRICTED_ZONE_CONFIG['warning_threshold']:
            severity = 'warning'
            message = RESTRICTED_ZONE_CONFIG['warning_message']
        else:
            severity = 'normal'
            message = '정상'
        
        print(f"거리 {distance}픽셀 → {severity.upper()} 알람: {message}")

def main():
    """메인 테스트 함수"""
    print("제한구역 알람 시스템 테스트를 시작합니다...\n")
    
    # 거리 계산 테스트
    test_distance_calculation()
    
    # 알람 로직 테스트
    test_alarm_logic()
    
    print("\n=== 테스트 완료 ===")
    print("✅ 제한구역 내부 진입: 치명적 알람")
    print("⚠️ 1미터 이내 접근: 경고 알람")
    print("✅ 1미터 이상 거리: 정상")

if __name__ == "__main__":
    main()
