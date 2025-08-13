import csv
import time
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class PerformanceLogger:
    """성능 측정과 CSV 저장을 담당하는 간소화된 클래스"""
    
    def __init__(self, output_dir: str = "result"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # CSV 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = self.output_dir / f"performance_log_{timestamp}.csv"
        
        # CSV 헤더
        self.csv_headers = [
            'frame_id', 'object_count', 'detection_time_ms', 'tracking_time_ms',
            'pre_match_time_ms', 'same_camera_time_ms', 'cross_camera_time_ms', 'total_time_ms'
        ]
        
        # 데이터 저장소
        self.memory_data = []
        self.current_frame_data = {}
        
        # CSV 초기화
        self._initialize_csv()
        
        print(f"📊 Performance logger initialized: {self.csv_filename}")
    
    def _initialize_csv(self):
        """CSV 파일 초기화"""
        with open(self.csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.csv_headers)
    
    def start_frame_timing(self, frame_id: int, camera_id: int):
        """프레임별 타이밍 시작"""
        self.current_frame_data = {
            'frame_id': frame_id, 'camera_id': camera_id, 'start_time': time.time(),
            'detection_start': None, 'detection_end': None, 'tracking_start': None,
            'tracking_end': None, 'pre_match_start': None, 'pre_match_end': None,
            'same_camera_reid_start': None, 'same_camera_reid_end': None,
            'cross_camera_reid_start': None, 'cross_camera_reid_end': None, 'object_count': 0
        }
    
    def start_detection_timing(self):
        """탐지 타이밍 시작"""
        self.current_frame_data['detection_start'] = time.time()
    
    def end_detection_timing(self):
        """탐지 타이밍 종료"""
        self.current_frame_data['detection_end'] = time.time()
    
    def start_tracking_timing(self):
        """트래킹 타이밍 시작"""
        self.current_frame_data['tracking_start'] = time.time()
    
    def end_tracking_timing(self):
        """트래킹 타이밍 종료"""
        self.current_frame_data['tracking_end'] = time.time()
    
    def start_pre_match_timing(self):
        """사전 등록 매칭 타이밍 시작"""
        self.current_frame_data['pre_match_start'] = time.time()
    
    def end_pre_match_timing(self):
        """사전 등록 매칭 타이밍 종료"""
        self.current_frame_data['pre_match_end'] = time.time()
    
    def start_same_camera_reid_timing(self):
        """같은 카메라 내 ReID 타이밍 시작"""
        self.current_frame_data['same_camera_reid_start'] = time.time()
    
    def end_same_camera_reid_timing(self):
        """같은 카메라 내 ReID 타이밍 종료"""
        self.current_frame_data['same_camera_reid_end'] = time.time()
    
    def start_cross_camera_reid_timing(self):
        """다른 카메라 간 ReID 타이밍 시작"""
        self.current_frame_data['cross_camera_reid_start'] = time.time()
    
    def end_cross_camera_reid_timing(self):
        """다른 카메라 간 ReID 타이밍 종료"""
        self.current_frame_data['cross_camera_reid_end'] = time.time()
    
    def set_object_count(self, count: int):
        """객체 수 설정"""
        self.current_frame_data['object_count'] = count
    
    def log_frame_performance(self):
        """현재 프레임의 성능 데이터를 CSV에 저장"""
        if not self.current_frame_data:
            return
        
        # 시간 계산
        detection_time = self._calculate_time('detection_start', 'detection_end')
        tracking_time = self._calculate_time('tracking_start', 'tracking_end')
        pre_match_time = self._calculate_time('pre_match_start', 'pre_match_end')
        same_camera_reid_time = self._calculate_time('same_camera_reid_start', 'same_camera_reid_end')
        cross_camera_reid_time = self._calculate_time('cross_camera_reid_start', 'cross_camera_reid_end')
        total_time = (time.time() - self.current_frame_data['start_time']) * 1000
        
        # 데이터 준비
        row_data = [
            self.current_frame_data['frame_id'], self.current_frame_data['object_count'],
            round(detection_time, 2), round(tracking_time, 2), round(pre_match_time, 2),
            round(same_camera_reid_time, 2), round(cross_camera_reid_time, 2),
            round(total_time, 2)
        ]
        
        # CSV에 저장
        self._write_to_csv(row_data)
    
    def _calculate_time(self, start_key: str, end_key: str) -> float:
        """시간 차이 계산 (ms)"""
        start = self.current_frame_data.get(start_key)
        end = self.current_frame_data.get(end_key)
        if start and end:
            return (end - start) * 1000
        return 0
    
    def _write_to_csv(self, row_data: List):
        """CSV에 데이터 저장"""
        with open(self.csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row_data)
    
    def print_summary(self):
        """성능 요약 출력"""
        if not os.path.exists(self.csv_filename):
            print("📊 저장된 성능 데이터가 없습니다.")
            return
        
        # CSV 데이터 읽기
        data_source = self._read_csv_data()
        
        if not data_source:
            print("📊 CSV 파일에 데이터가 없습니다.")
            return
        
        # 기본 통계 계산
        basic_stats = self._calculate_basic_stats(data_source)
        reid_stats = self._calculate_reid_stats(data_source)
        
        # 요약 출력
        print(f"\n📊 성능 요약 (총 {basic_stats['frame_count']} 프레임)")
        print("=" * 60)
        print(f"총 객체 수: {basic_stats['object_count']}")
        print(f"평균 탐지 시간: {basic_stats['avg_detection']:.2f}ms")
        print(f"평균 트래킹 시간: {basic_stats['avg_tracking']:.2f}ms")
        print(f"평균 총 시간: {basic_stats['avg_total']:.2f}ms")
        
        print(f"\n📊 ReID 통계 (0값 제외)")
        print("-" * 40)
        for category, data in reid_stats.items():
            if data['frame_count'] > 0:
                print(f"{category:15}: {data['frame_count']:3d} 프레임, "
                      f"객체 {data['object_count']:3d}개, "
                      f"평균 {data['avg_time']:6.2f}ms")
        
        print(f"\n📁 CSV 파일 저장 완료: {self.csv_filename}")
    
    def _read_csv_data(self) -> List:
        """CSV 파일에서 데이터 읽기"""
        data = []
        with open(self.csv_filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # 헤더 건너뛰기
            for row in reader:
                data.append(row)
        return data
    
    def _calculate_basic_stats(self, data_source: List) -> Dict:
        """기본 통계 계산"""
        detection_times = [float(row[2]) for row in data_source]
        tracking_times = [float(row[3]) for row in data_source]
        total_times = [float(row[7]) for row in data_source]
        object_counts = [int(row[1]) for row in data_source]
        
        return {
            'object_count': sum(object_counts),
            'frame_count': len(data_source),
            'avg_detection': sum(detection_times) / len(detection_times),
            'avg_tracking': sum(tracking_times) / len(tracking_times),
            'avg_total': sum(total_times) / len(total_times)
        }
    
    def _calculate_reid_stats(self, data_source: List) -> Dict:
        """ReID 관련 통계 계산 (0값 제외)"""
        # 0이 아닌 값들만 필터링
        pre_match_times = [float(row[4]) for row in data_source if float(row[4]) > 0]
        same_camera_times = [float(row[5]) for row in data_source if float(row[5]) > 0]
        cross_camera_times = [float(row[6]) for row in data_source if float(row[6]) > 0]
        
        # 각 카테고리별로 0이 아닌 프레임 수 계산
        pre_match_frames = len([row for row in data_source if float(row[4]) > 0])
        same_camera_frames = len([row for row in data_source if float(row[5]) > 0])
        cross_camera_frames = len([row for row in data_source if float(row[6]) > 0])
        
        # 각 카테고리별로 0이 아닌 프레임의 객체 수 합계
        pre_match_objects = sum([int(row[1]) for row in data_source if float(row[4]) > 0])
        same_camera_objects = sum([int(row[1]) for row in data_source if float(row[5]) > 0])
        cross_camera_objects = sum([int(row[1]) for row in data_source if float(row[6]) > 0])
        
        def calc_avg_time(times_list):
            return sum(times_list) / len(times_list) if times_list else 0
        
        return {
            'pre_match': {
                'object_count': pre_match_objects,
                'frame_count': pre_match_frames,
                'avg_time': calc_avg_time(pre_match_times)
            },
            'same_camera': {
                'object_count': same_camera_objects,
                'frame_count': same_camera_frames,
                'avg_time': calc_avg_time(same_camera_times)
            },
            'cross_camera': {
                'object_count': cross_camera_objects,
                'frame_count': cross_camera_frames,
                'avg_time': calc_avg_time(cross_camera_times)
            }
        }
