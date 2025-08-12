import csv
import time
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class PerformanceLogger:
    """
    각 구간별 성능 측정과 CSV 저장을 담당하는 클래스
    """
    
    def __init__(self, output_dir: str = "result"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # CSV 파일명 생성 (타임스탬프 포함)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = self.output_dir / f"performance_log_{timestamp}.csv"
        
        # 객체 개수별 통계를 위한 별도 CSV 파일
        self.object_count_csv_filename = self.output_dir / f"object_count_stats_{timestamp}.csv"
        
        # CSV 헤더 정의
        self.csv_headers = [
            'frame_id',
            'camera_id', 
            'object_count',
            'detection_time_ms',
            'tracking_time_ms',
            'same_camera_reid_time_ms',
            'cross_camera_reid_time_ms',
            'total_time_ms',
            'timestamp'
        ]
        
        # CSV 파일 초기화
        self._initialize_csv()
        
        # 성능 측정을 위한 임시 저장소
        self.current_frame_data = {}
        
    def _initialize_csv(self):
        """CSV 파일을 생성하고 헤더를 작성합니다."""
        with open(self.csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.csv_headers)
        
        # 객체 개수별 통계 CSV 파일 초기화
        object_count_headers = [
            'object_count',
            'frame_count',
            'avg_detection_time_ms',
            'avg_tracking_time_ms',
            'avg_same_camera_reid_time_ms',
            'avg_cross_camera_reid_time_ms',
            'avg_total_time_ms',
            'min_detection_time_ms',
            'max_detection_time_ms',
            'min_tracking_time_ms',
            'max_tracking_time_ms',
            'min_total_time_ms',
            'max_total_time_ms'
        ]
        
        with open(self.object_count_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(object_count_headers)
        
        print(f"📊 Performance logger initialized: {self.csv_filename}")
        print(f"📊 Object count stats will be saved to: {self.object_count_csv_filename}")
    
    def start_frame_timing(self, frame_id: int, camera_id: int):
        """프레임별 타이밍을 시작합니다."""
        self.current_frame_data = {
            'frame_id': frame_id,
            'camera_id': camera_id,
            'start_time': time.time(),
            'detection_start': None,
            'detection_end': None,
            'tracking_start': None,
            'tracking_end': None,
            'same_camera_reid_start': None,
            'same_camera_reid_end': None,
            'cross_camera_reid_start': None,
            'cross_camera_reid_end': None,
            'object_count': 0
        }
    
    def start_detection_timing(self):
        """탐지 타이밍을 시작합니다."""
        self.current_frame_data['detection_start'] = time.time()
    
    def end_detection_timing(self):
        """탐지 타이밍을 종료합니다."""
        self.current_frame_data['detection_end'] = time.time()
    
    def start_tracking_timing(self):
        """트래킹 타이밍을 시작합니다."""
        self.current_frame_data['tracking_start'] = time.time()
    
    def end_tracking_timing(self):
        """트래킹 타이밍을 종료합니다."""
        self.current_frame_data['tracking_end'] = time.time()
    
    def start_same_camera_reid_timing(self):
        """같은 카메라 내 ReID 타이밍을 시작합니다."""
        self.current_frame_data['same_camera_reid_start'] = time.time()
    
    def end_same_camera_reid_timing(self):
        """같은 카메라 내 ReID 타이밍을 종료합니다."""
        self.current_frame_data['same_camera_reid_end'] = time.time()
    
    def start_cross_camera_reid_timing(self):
        """다른 카메라 간 ReID 타이밍을 시작합니다."""
        self.current_frame_data['cross_camera_reid_start'] = time.time()
    
    def end_cross_camera_reid_timing(self):
        """다른 카메라 간 ReID 타이밍을 종료합니다."""
        self.current_frame_data['cross_camera_reid_end'] = time.time()
    
    def set_object_count(self, count: int):
        """객체 수를 설정합니다."""
        self.current_frame_data['object_count'] = count
    
    def log_frame_performance(self):
        """현재 프레임의 성능 데이터를 CSV에 저장합니다."""
        if not self.current_frame_data:
            return
        
        # 시간 계산
        detection_time = 0
        if self.current_frame_data['detection_start'] and self.current_frame_data['detection_end']:
            detection_time = (self.current_frame_data['detection_end'] - self.current_frame_data['detection_start']) * 1000
        
        tracking_time = 0
        if self.current_frame_data['tracking_start'] and self.current_frame_data['tracking_end']:
            tracking_time = (self.current_frame_data['tracking_end'] - self.current_frame_data['tracking_start']) * 1000
        
        same_camera_reid_time = 0
        if self.current_frame_data['same_camera_reid_start'] and self.current_frame_data['same_camera_reid_end']:
            same_camera_reid_time = (self.current_frame_data['same_camera_reid_end'] - self.current_frame_data['same_camera_reid_start']) * 1000
        
        cross_camera_reid_time = 0
        if self.current_frame_data['cross_camera_reid_start'] and self.current_frame_data['cross_camera_reid_end']:
            cross_camera_reid_time = (self.current_frame_data['cross_camera_reid_end'] - self.current_frame_data['cross_camera_reid_start']) * 1000
        
        total_time = (time.time() - self.current_frame_data['start_time']) * 1000
        
        # CSV에 데이터 저장
        row_data = [
            self.current_frame_data['frame_id'],
            self.current_frame_data['camera_id'],
            self.current_frame_data['object_count'],
            round(detection_time, 2),
            round(tracking_time, 2),
            round(same_camera_reid_time, 2),
            round(cross_camera_reid_time, 2),
            round(total_time, 2),
            datetime.now().isoformat()
        ]
        
        with open(self.csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row_data)
        
        # 디버그 출력
        print(f"[PERF] Frame {self.current_frame_data['frame_id']} (Cam {self.current_frame_data['camera_id']}): "
              f"Detection={detection_time:.2f}ms, Tracking={tracking_time:.2f}ms, "
              f"SameCam ReID={same_camera_reid_time:.2f}ms, CrossCam ReID={cross_camera_reid_time:.2f}ms, "
              f"Total={total_time:.2f}ms, Objects={self.current_frame_data['object_count']}")
    
    def get_summary_stats(self) -> Dict:
        """전체 성능 통계를 계산합니다."""
        if not os.path.exists(self.csv_filename):
            return {}
        
        detection_times = []
        tracking_times = []
        same_camera_reid_times = []
        cross_camera_reid_times = []
        total_times = []
        object_counts = []
        
        with open(self.csv_filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                detection_times.append(float(row['detection_time_ms']))
                tracking_times.append(float(row['tracking_time_ms']))
                same_camera_reid_times.append(float(row['same_camera_reid_time_ms']))
                cross_camera_reid_times.append(float(row['cross_camera_reid_time_ms']))
                total_times.append(float(row['total_time_ms']))
                object_counts.append(int(row['object_count']))
        
        if not detection_times:
            return {}
        
        def calculate_stats(values):
            return {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'total': sum(values)
            }
        
        return {
            'detection': calculate_stats(detection_times),
            'tracking': calculate_stats(tracking_times),
            'same_camera_reid': calculate_stats(same_camera_reid_times),
            'cross_camera_reid': calculate_stats(cross_camera_reid_times),
            'total': calculate_stats(total_times),
            'object_count': calculate_stats(object_counts),
            'total_frames': len(detection_times)
        }
    
    def print_summary(self):
        """성능 요약을 출력합니다."""
        stats = self.get_summary_stats()
        if not stats:
            print("📊 No performance data available")
            return
        
        print("\n" + "="*60)
        print("📊 PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Total Frames Processed: {stats['total_frames']}")
        print(f"CSV File: {self.csv_filename}")
        print("\nAverage Times (ms):")
        print(f"  Detection: {stats['detection']['mean']:.2f} (min: {stats['detection']['min']:.2f}, max: {stats['detection']['max']:.2f})")
        print(f"  Tracking: {stats['tracking']['mean']:.2f} (min: {stats['tracking']['min']:.2f}, max: {stats['tracking']['max']:.2f})")
        print(f"  Same Camera ReID: {stats['same_camera_reid']['mean']:.2f} (min: {stats['same_camera_reid']['min']:.2f}, max: {stats['same_camera_reid']['max']:.2f})")
        print(f"  Cross Camera ReID: {stats['cross_camera_reid']['mean']:.2f} (min: {stats['cross_camera_reid']['min']:.2f}, max: {stats['cross_camera_reid']['max']:.2f})")
        print(f"  Total per Frame: {stats['total']['mean']:.2f} (min: {stats['total']['min']:.2f}, max: {stats['total']['max']:.2f})")
        print(f"  Average Objects per Frame: {stats['object_count']['mean']:.1f}")
        print("="*60)
        
        # 객체 개수별 통계 생성 및 저장
        self.generate_object_count_stats()
    
    def generate_object_count_stats(self):
        """객체 개수별 성능 통계를 생성하고 CSV에 저장합니다."""
        if not os.path.exists(self.csv_filename):
            print("📊 No performance data available for object count analysis")
            return
        
        # 객체 개수별로 데이터 그룹화
        object_count_data = {}
        
        with open(self.csv_filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                object_count = int(row['object_count'])
                if object_count not in object_count_data:
                    object_count_data[object_count] = {
                        'detection_times': [],
                        'tracking_times': [],
                        'same_camera_reid_times': [],
                        'cross_camera_reid_times': [],
                        'total_times': []
                    }
                
                object_count_data[object_count]['detection_times'].append(float(row['detection_time_ms']))
                object_count_data[object_count]['tracking_times'].append(float(row['tracking_time_ms']))
                object_count_data[object_count]['same_camera_reid_times'].append(float(row['same_camera_reid_time_ms']))
                object_count_data[object_count]['cross_camera_reid_times'].append(float(row['cross_camera_reid_time_ms']))
                object_count_data[object_count]['total_times'].append(float(row['total_time_ms']))
        
        # 객체 개수별 통계 계산 및 CSV 저장
        with open(self.object_count_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'object_count',
                'frame_count',
                'avg_detection_time_ms',
                'avg_tracking_time_ms',
                'avg_same_camera_reid_time_ms',
                'avg_cross_camera_reid_time_ms',
                'avg_total_time_ms',
                'min_detection_time_ms',
                'max_detection_time_ms',
                'min_tracking_time_ms',
                'max_tracking_time_ms',
                'min_total_time_ms',
                'max_total_time_ms'
            ])
            
            # 객체 개수 순으로 정렬
            for object_count in sorted(object_count_data.keys()):
                data = object_count_data[object_count]
                frame_count = len(data['detection_times'])
                
                # 평균 계산
                avg_detection = sum(data['detection_times']) / frame_count
                avg_tracking = sum(data['tracking_times']) / frame_count
                avg_same_camera_reid = sum(data['same_camera_reid_times']) / frame_count
                avg_cross_camera_reid = sum(data['cross_camera_reid_times']) / frame_count
                avg_total = sum(data['total_times']) / frame_count
                
                # 최소/최대 계산
                min_detection = min(data['detection_times'])
                max_detection = max(data['detection_times'])
                min_tracking = min(data['tracking_times'])
                max_tracking = max(data['tracking_times'])
                min_total = min(data['total_times'])
                max_total = max(data['total_times'])
                
                writer.writerow([
                    object_count,
                    frame_count,
                    round(avg_detection, 2),
                    round(avg_tracking, 2),
                    round(avg_same_camera_reid, 2),
                    round(avg_cross_camera_reid, 2),
                    round(avg_total, 2),
                    round(min_detection, 2),
                    round(max_detection, 2),
                    round(min_tracking, 2),
                    round(max_tracking, 2),
                    round(min_total, 2),
                    round(max_total, 2)
                ])
        
        # 객체 개수별 통계 출력
        print(f"\n📊 OBJECT COUNT STATISTICS")
        print("="*60)
        print(f"Object count stats saved to: {self.object_count_csv_filename}")
        print("\nObject Count Breakdown:")
        
        for object_count in sorted(object_count_data.keys()):
            data = object_count_data[object_count]
            frame_count = len(data['detection_times'])
            avg_total = sum(data['total_times']) / frame_count
            avg_detection = sum(data['detection_times']) / frame_count
            avg_tracking = sum(data['tracking_times']) / frame_count
            
            print(f"  {object_count} objects: {frame_count} frames, "
                  f"Avg Total: {avg_total:.2f}ms, "
                  f"Avg Detection: {avg_detection:.2f}ms, "
                  f"Avg Tracking: {avg_tracking:.2f}ms")
        
        print("="*60)
