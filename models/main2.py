import asyncio
from datetime import datetime, timezone
import requests
import json
import sys
import os
import torch
import time
import numpy as np

# 프로젝트 루트 경로 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.extend([
    project_root,
    os.path.join(project_root, 'ByteTrack'),
    os.path.join(project_root, 'deep-person-reid-master'),
    os.path.join(project_root, 'models', 'mapping')  # point_transformer 경로
])

# Node.js 서버 URL (기존 main.py 형식)
WORKER_URL = "http://localhost:5000/workers"
ZONE_URL = "http://localhost:5000/zones"
VIOLATION_URL = "http://localhost:5000/violations"

# Redis Global ReID 모듈 import
try:
    from redis_global_reid_main_v2 import run_tracking_realtime, FeatureExtractor, RedisGlobalReIDManagerV2
    import argparse
except ImportError as e:
    print(f"Redis Global ReID 모듈 import 실패: {e}")
    sys.exit(1)

# PPE 탐지 모듈 import
try:
    from ppe_detector import PPEDetector, ViolationHistory
except ImportError as e:
    print(f"PPE 탐지 모듈 import 실패: {e}")
    sys.exit(1)

# ReID 추적 실행을 위한 인자 설정
parser = argparse.ArgumentParser()
parser.add_argument('--videos', nargs='+', type=str, 
                   default=["../test_video/KSEB03.mp4"], 
                   help='List of video file paths.')
parser.add_argument('--yolo_model', type=str, 
                   default="models/weights/bestcctv.pt", 
                   help='Path to the YOLOv11 model file.')
parser.add_argument('--ppe_model', type=str,
                   default="models/weights/best_yolo11n.pt",
                   help='Path to the PPE detection model file.')
parser.add_argument('--redis_host', type=str, default="localhost", help='Redis server host.')
parser.add_argument('--redis_port', type=int, default=6379, help='Redis server port.')

args = parser.parse_args([])  # 빈 리스트로 기본값 사용

# 제한구역 설정 파일 import
def load_zone_config():
    """제한구역 설정 로드 (지정된 JSON 파일 우선, 자동 검색 차선)"""
    global RESTRICTED_ZONE_CONFIG, RESTRICTED_ZONE_EXAMPLES, CURRENT_ZONE_TYPE
    
    # 1. 지정된 JSON 파일에서 좌표 로드 시도
    specified_json = os.path.join(project_root, "zone_coordinates_20250808_162417.json")
    if os.path.exists(specified_json):
        try:
            with open(specified_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                zones = data.get('zones', [])
                if zones:
                    # JSON에서 설정 생성
                    RESTRICTED_ZONE_CONFIG = {
                        'severity': 'critical',
                        'alarm_threshold': 0,  # 제한구역 내부 진입 시 치명적 알람
                        'warning_threshold': 100,  # 1미터(100픽셀) 이내 접근 시 경고 알람
                        'alarm_message': '제한구역 내부 진입! 즉시 이탈하세요!',
                        'warning_message': '제한구역 1미터 이내 접근! 주의하세요!',
                        'color': 'red'
                    }
                    
                    RESTRICTED_ZONE_EXAMPLES = {}
                    for i, zone in enumerate(zones):
                        zone_name = zone.get('zone_name', f'zone_{i+1}')
                        RESTRICTED_ZONE_EXAMPLES[zone_name] = {
                            'x1': zone['x1'],
                            'y1': zone['y1'],
                            'x2': zone['x2'],
                            'y2': zone['y2'],
                            'threshold': zone['threshold']
                        }
                    
                    CURRENT_ZONE_TYPE = list(RESTRICTED_ZONE_EXAMPLES.keys())[0]
                    print(f"[CONFIG] 지정된 JSON 파일에서 제한구역 설정 로드 완료: {specified_json}")
                    print(f"[CONFIG] 로드된 제한구역: {list(RESTRICTED_ZONE_EXAMPLES.keys())}")
                    return True
        except Exception as e:
            print(f"[WARNING] 지정된 JSON 파일 로드 실패: {e}")
    
    # 2. 자동으로 최신 JSON 파일 검색
    json_files = [f for f in os.listdir('.') if f.startswith('zone_coordinates_') and f.endswith('.json')]
    if json_files:
        # 가장 최근 JSON 파일 선택
        latest_json = sorted(json_files)[-1]
        try:
            with open(latest_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                zones = data.get('zones', [])
                if zones:
                    # JSON에서 설정 생성
                    RESTRICTED_ZONE_CONFIG = {
                        'severity': 'critical',
                        'alarm_threshold': 0,  # 제한구역 내부 진입 시 치명적 알람
                        'warning_threshold': 100,  # 1미터(100픽셀) 이내 접근 시 경고 알람
                        'alarm_message': '제한구역 내부 진입! 즉시 이탈하세요!',
                        'warning_message': '제한구역 1미터 이내 접근! 주의하세요!',
                        'color': 'red'
                    }
                    
                    RESTRICTED_ZONE_EXAMPLES = {}
                    for i, zone in enumerate(zones):
                        zone_name = zone.get('zone_name', f'zone_{i+1}')
                        RESTRICTED_ZONE_EXAMPLES[zone_name] = {
                            'x1': zone['x1'],
                            'y1': zone['y1'],
                            'x2': zone['x2'],
                            'y2': zone['y2'],
                            'threshold': zone['threshold']
                        }
                    
                    CURRENT_ZONE_TYPE = list(RESTRICTED_ZONE_EXAMPLES.keys())[0]
                    print(f"[CONFIG] 자동 검색 JSON 파일에서 제한구역 설정 로드 완료: {latest_json}")
                    print(f"[CONFIG] 로드된 제한구역: {list(RESTRICTED_ZONE_EXAMPLES.keys())}")
                    return True
        except Exception as e:
            print(f"[WARNING] 자동 검색 JSON 파일 로드 실패: {e}")
    
    # 3. 설정 파일에서 로드 시도
    try:
        from restricted_zone_config import (
            RESTRICTED_ZONE_CONFIG, 
            RESTRICTED_ZONE_EXAMPLES, 
            CURRENT_ZONE_TYPE,
            set_zone_type,
            get_zone_coordinates,
            list_available_zones
        )
        print(f"[CONFIG] 제한구역 설정 파일 로드 완료")
        return True
    except ImportError as e:
        print(f"[WARNING] 제한구역 설정 파일을 찾을 수 없음: {e}")
        # 기본 설정
        RESTRICTED_ZONE_CONFIG = {
            'severity': 'critical',
            'alarm_threshold': 0,  # 제한구역 내부 진입 시 치명적 알람
            'warning_threshold': 100,  # 1미터(100픽셀) 이내 접근 시 경고 알람
            'alarm_message': '제한구역 내부 진입! 즉시 이탈하세요!',
            'warning_message': '제한구역 1미터 이내 접근! 주의하세요!',
            'color': 'red'
        }
        RESTRICTED_ZONE_EXAMPLES = {
            'center_zone': {'x1': 800, 'y1': 400, 'x2': 1120, 'y2': 680, 'threshold': 100}
        }
        CURRENT_ZONE_TYPE = 'center_zone'
        return False

# 설정 로드
load_zone_config()

# 통합된 위험구역 감지 및 알람 시스템
class HazardZoneDetector:
    def __init__(self, zone_type='center_zone'):
        """
        제한구역 감지기 초기화
        Args:
            zone_type: 사용할 제한구역 타입 ('center_zone', 'right_zone', 'bottom_zone', 'machine_zone')
        """
        # 제한구역 정의 - RESTRICTED_ZONE_EXAMPLES에서 선택하거나 직접 설정
        if zone_type in RESTRICTED_ZONE_EXAMPLES:
            self.restricted_zone = RESTRICTED_ZONE_EXAMPLES[zone_type].copy()
        else:
            # 기본 제한구역 (화면 중앙)
            self.restricted_zone = {
                'x1': 800, 'y1': 400,   # 좌상단
                'x2': 1120, 'y2': 680,  # 우하단
                'threshold': 100         # 감지 반경
            }
        
        print(f"[INFO] 제한구역 설정: {zone_type}")
        print(f"[INFO] 좌표: ({self.restricted_zone['x1']},{self.restricted_zone['y1']}) ~ ({self.restricted_zone['x2']},{self.restricted_zone['y2']})")
        print(f"[INFO] 감지 반경: {self.restricted_zone['threshold']} 픽셀")
        
        # 알람 시스템 관련 변수들
        self.active_alarms = {}  # {worker_id: {zone_id, start_time, severity, alarm_type}}
        self.alarm_history = []
        self.alarm_cooldown = 30  # 30초 쿨다운 (중복 알람 방지)
        self.last_alarm_time = {}  # {worker_id_zone_id: timestamp}
   
    def calculate_distance(self, point, zone_coords):
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
   
    def check_restricted_zone_proximity(self, worker_position):
        """워커가 제한구역에 가까워졌는지 확인하고 알람 처리"""
        violations = []
        current_time = time.time()
        
        distance = self.calculate_distance(worker_position, self.restricted_zone)
        warning_threshold = RESTRICTED_ZONE_CONFIG['warning_threshold']
        
        # 제한구역 내부 진입 (distance = 0) 또는 경고 임계값 이내 접근
        if distance <= warning_threshold:
            # 제한구역 내부 진입 시 치명적 알람
            if distance == 0:
                severity = 'critical'
            # 경고 임계값 이내 접근 시 경고 알람
            else:
                severity = 'warning'
            
            violation = {
                'worker_id': worker_position['worker_id'],
                'zone_id': 'restricted_zone',
                'zone_type': 'restricted',
                'distance': float(distance),
                'threshold': warning_threshold,
                'timestamp': datetime.now().isoformat(),
                'severity': severity
            }
            violations.append(violation)
            
            # 알람 처리
            self._process_alarm(violation, current_time)
        
        return violations
    
    def _process_alarm(self, violation, current_time):
        """위반에 대한 알람 처리"""
        worker_id = violation['worker_id']
        zone_id = violation['zone_id']
        distance = violation['distance']
        severity = violation['severity']
        
        # 쿨다운 체크
        alarm_key = f"{worker_id}_{zone_id}_{severity}"
        if alarm_key in self.last_alarm_time:
            if current_time - self.last_alarm_time[alarm_key] < self.alarm_cooldown:
                return  # 쿨다운 중이면 스킵
        
        # 알람 트리거 (severity에 따라)
        if severity == 'critical':
            self._trigger_critical_alarm(worker_id, zone_id, distance, current_time)
        elif severity == 'warning':
            self._trigger_warning_alarm(worker_id, zone_id, distance, current_time)
    
    def _trigger_critical_alarm(self, worker_id, zone_id, distance, timestamp):
        """치명적 알람 트리거"""
        alarm_data = {
            'type': 'critical',
            'worker_id': worker_id,
            'zone_id': zone_id,
            'zone_type': 'restricted',
            'distance': distance,
            'message': RESTRICTED_ZONE_CONFIG['alarm_message'],
            'timestamp': datetime.fromtimestamp(timestamp).isoformat(),
            'requires_immediate_action': True,
            'severity': 'critical',
            'color': RESTRICTED_ZONE_CONFIG['color']
        }
        self._send_alarm_to_backend(alarm_data)
        self._update_alarm_status(worker_id, zone_id, 'critical', timestamp)
    
    def _trigger_warning_alarm(self, worker_id, zone_id, distance, timestamp):
        """경고 알람 트리거"""
        alarm_data = {
            'type': 'warning',
            'worker_id': worker_id,
            'zone_id': zone_id,
            'zone_type': 'restricted',
            'distance': distance,
            'message': RESTRICTED_ZONE_CONFIG['warning_message'],
            'timestamp': datetime.fromtimestamp(timestamp).isoformat(),
            'requires_immediate_action': False,
            'severity': 'warning',
            'color': RESTRICTED_ZONE_CONFIG['color']
        }
        self._send_alarm_to_backend(alarm_data)
        self._update_alarm_status(worker_id, zone_id, 'warning', timestamp)
    
    def _update_alarm_status(self, worker_id, zone_id, alarm_type, timestamp):
        """알람 상태 업데이트"""
        alarm_key = f"{worker_id}_{zone_id}"
        self.active_alarms[alarm_key] = {
            'worker_id': worker_id,
            'zone_id': zone_id,
            'alarm_type': alarm_type,
            'start_time': timestamp,
            'last_triggered': timestamp
        }
        self.last_alarm_time[alarm_key] = timestamp
    
    def clear_alarm(self, worker_id, zone_id):
        """알람 해제 (위험구역 이탈 시)"""
        alarm_key = f"{worker_id}_{zone_id}"
        if alarm_key in self.active_alarms:
            alarm_info = self.active_alarms.pop(alarm_key)
            # 알람 해제 이벤트 전송
            clear_data = {
                'type': 'clear',
                'worker_id': worker_id,
                'zone_id': zone_id,
                'message': f"{worker_id}가 {zone_id}에서 이탈했습니다.",
                'timestamp': datetime.now().isoformat(),
                'duration': time.time() - alarm_info['start_time']
            }
            self._send_alarm_to_backend(clear_data)
    
    def _send_alarm_to_backend(self, alarm_data):
        """백엔드로 알람 전송"""
        try:
            response = requests.post(
                "http://localhost:5000/alerts",
                json={"alerts": [alarm_data]},
                timeout=2
            )
            if response.status_code == 200:
                print(f"[ALARM] {alarm_data['type'].upper()} - {alarm_data['worker_id']} in {alarm_data['zone_id']} ({alarm_data['distance']:.1f}m)")
            else:
                print(f"[ERROR] 알람 전송 실패: {response.status_code}")
        except Exception as e:
            print(f"[ERROR] 알람 전송 실패: {e}")
    
    def get_active_alarms(self):
        """현재 활성 알람 목록 반환"""
        return list(self.active_alarms.values())
    
    def get_alarm_history(self, limit=100):
        """알람 히스토리 반환"""
        return self.alarm_history[-limit:] if self.alarm_history else []

# 전역 변수로 모델들 초기화
reid_extractor = None
global_reid_manager = None
tracking_generators = []
ppe_detector = None
hazard_detector = None

def initialize_models():
    """모든 모델들을 초기화"""
    global reid_extractor, global_reid_manager, tracking_generators, ppe_detector, hazard_detector
    
    try:
        # ReID 모델 초기화
        reid_extractor = FeatureExtractor(
            model_name='osnet_ibn_x1_0',
            model_path=None,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        global_reid_manager = RedisGlobalReIDManagerV2(
            similarity_threshold=0.5,
            feature_ttl=3000,
            max_features_per_camera=10,
            redis_host='localhost',
            redis_port=6379,
            frame_rate=30
        )
        
        # 실시간 추적 제너레이터 생성
        tracking_generators = []
        for i, video_path in enumerate(args.videos):
            generator = run_tracking_realtime(
                video_path, 
                args.yolo_model, 
                reid_extractor, 
                camera_id=i, 
                global_reid_manager=global_reid_manager
            )
            tracking_generators.append(generator)
        
        # PPE 탐지 모델 초기화
        if os.path.exists(args.ppe_model):
            detection_items = {
                'detect_no_safety_vest_or_helmet': True,
                'detect_near_machinery_or_vehicle': True,
                'detect_in_restricted_area': False
            }
            ppe_detector = PPEDetector(args.ppe_model, detection_items)
            print(f"✅ PPE 모델 로드 성공: {args.ppe_model}")
        else:
            print(f"⚠️ PPE 모델 파일을 찾을 수 없음: {args.ppe_model}")
            ppe_detector = None
        
        # 위험구역 감지 및 알람 시스템 초기화
        # 설정 파일에서 제한구역 타입 가져오기
        hazard_detector = HazardZoneDetector(zone_type=CURRENT_ZONE_TYPE)
        print(f"[INFO] 제한구역 감지 및 알람 시스템 초기화 완료")
        
        # 사용 가능한 제한구역 목록 출력
        try:
            list_available_zones()
        except:
            pass
        
        print(f"[INFO] Initialized {len(tracking_generators)} video trackers")
        return True
    
    except Exception as e:
        print(f"모델 초기화 실패: {e}")
        return False

def detect_ppe_violations(frame, detections):
    """PPE 위반 탐지"""
    if ppe_detector is None:
        return []
    
    try:
        # PPE 탐지 수행
        ppe_detections = ppe_detector.detect_frame(frame)
        
        # 위반 분석
        violations = ppe_detector.analyze_safety_violations(ppe_detections)
        
        # 위반 결과를 워커별로 매핑
        ppe_violations = []
        for detection in detections:
            worker_id = f"W{detection['workerID']:03d}"
            
            # 워커 위치에서 PPE 위반 확인
            worker_center_x = detection['position_X']
            worker_center_y = detection['position_Y']
            
            # PPE 위반이 있는지 확인 (간단한 위치 기반 매핑)
            for ppe_det in ppe_detections:
                if ppe_det['class_name'] in ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']:
                    ppe_center_x = (ppe_det['bbox'][0] + ppe_det['bbox'][2]) / 2
                    ppe_center_y = (ppe_det['bbox'][1] + ppe_det['bbox'][3]) / 2
                    
                    # 워커와 PPE 위반 위치가 가까우면 해당 워커의 위반으로 간주
                    distance = np.sqrt((worker_center_x - ppe_center_x)**2 + (worker_center_y - ppe_center_y)**2)
                    if distance < 100:  # 임계값 조정 가능
                        ppe_violations.append({
                            'worker_id': worker_id,
                            'violation_type': ppe_det['class_name'],
                            'confidence': float(ppe_det['confidence']),
                            'timestamp': datetime.now().isoformat(),
                            'camera_id': detection['cameraID']
                        })
                        break
        
        return ppe_violations
        
    except Exception as e:
        print(f"PPE 위반 탐지 중 오류: {e}")
        return []

def run_detection():
    """실시간 ReID 추적 + PPE + 위험구역 감지"""
    start_time = time.time()  # 프레임 처리 시작 시간
    now = datetime.now(timezone.utc).isoformat()

    try:
        # 모든 카메라의 현재 프레임 결과 수집
        all_detections = []
        current_frames = []
        
        for i, generator in enumerate(tracking_generators):
            try:
                result = next(generator)  # 다음 프레임 결과 가져오기
                
                if isinstance(result, tuple):
                    detections, frame = result
                    all_detections.extend(detections)
                    current_frames.append(frame)
                else:
                    all_detections.extend(result)
                    current_frames.append(None)
                
            except StopIteration:
                # 비디오가 끝나면 다시 시작
                continue
        
        # PPE 위반 탐지 (실제 프레임 사용)
        ppe_violations = []
        if current_frames and current_frames[0] is not None:
            ppe_violations = detect_ppe_violations(current_frames[0], all_detections)
        
        # 제한구역 접근 탐지 (알람 자동 처리 포함)
        restricted_violations = []
        for detection in all_detections:
            worker_pos = {
                'worker_id': f"W{detection['workerID']:03d}",
                'x': detection['position_X'],
                'y': detection['position_Y']
            }
            violations = hazard_detector.check_restricted_zone_proximity(worker_pos)
            restricted_violations.extend(violations)
        
        # 감지 결과를 workers 형태로 변환
        workers = []
        for detection in all_detections:
            worker_id = f"W{detection['workerID']:03d}"
            
            # PPE 위반 확인
            ppe_violation = next((v for v in ppe_violations if v['worker_id'] == worker_id), None)
            
            # 제한구역 위반 확인
            restricted_violation = next((v for v in restricted_violations if v['worker_id'] == worker_id), None)
            
            status = "normal"
            if ppe_violation:
                status = "ppe_violation"
            elif restricted_violation:
                status = "restricted_zone_violation"
            
            worker = {
                "worker_id": worker_id,
                "x": float(detection['position_X']),  # float64 유지
                "y": float(detection['position_Y']),  # float64 유지
                "zone_id": f"Z{detection['cameraID']:02d}",
                "product_count": 1,
                "timestamp": now,
            }
            workers.append(worker)

        # 위반 정보 (PPE, ROI) - 기존 main.py 형식에 맞게 변환
        violations = []
        
        # PPE 위반을 기존 형식으로 변환
        for ppe_violation in ppe_violations:
            violation = {
                "worker_id": ppe_violation['worker_id'],
                "zone_id": f"Z{ppe_violation['camera_id']:02d}",
                "timestamp": now,
                "violations": {
                    "ppe": [ppe_violation['violation_type']],
                    "roi": []
                }
            }
            violations.append(violation)
        
        # 제한구역 위반을 기존 형식으로 변환
        for restricted_violation in restricted_violations:
            violation = {
                "worker_id": restricted_violation['worker_id'],
                "zone_id": restricted_violation['zone_id'],
                "timestamp": now,
                "violations": {
                    "ppe": [],
                    "roi": [restricted_violation['zone_type']]
                }
            }
            violations.append(violation)

        # zone 통계
        zone_stats = {}
        for w in workers:
            zid = w["zone_id"]
            zone_stats.setdefault(zid, {
                "zone_name": f"Zone {zid}",
                "zone_type": "작업구역",
                "total_product": 0,
                "active_workers": 0
            })
            zone_stats[zid]["total_product"] += w.get("product_count", 0)
            zone_stats[zid]["active_workers"] += 1

        zones = []
        for zid, stat in zone_stats.items():
            total = stat["total_product"]
            avg = 480 / total if total > 0 else None
            zones.append({
                "zone_id": zid,
                "zone_name": stat["zone_name"],
                "zone_type": stat["zone_type"],
                "timestamp": now,
                "active_workers": stat["active_workers"],
                "active_tasks": "",
                "avg_cycle_time_min": avg,
                "ppe_violations": sum(
                    1 for v in violations if v["zone_id"] == zid and v["violations"]["ppe"]
                ),
                "hazard_dwell_count": sum(
                    1 for v in violations if v["zone_id"] == zid and v["violations"]["roi"]
                ),
                "recent_alerts": ""
            })

        return {
            "workers": workers,
            "violations": violations,
            "zones": zones
        }
    
    except Exception as e:
        processing_time = time.time() - start_time  # 오류 시에도 처리 시간 계산
        print(f"ReID 추적 실행 중 오류: {e}")
        # 오류 발생 시 기본 데이터 반환
        return {
            "workers": [],
            "violations": [],
            "zones": [],
        }

# 주기적 전송 루프 (기존 main.py 형식)
async def detection_loop():
    print("[INFO] AI detection loop started")
    
    # 모델 초기화
    if not initialize_models():
        print("[ERROR] 모델 초기화 실패")
        return
    
    while True:
        result = run_detection()

        try:
            # 1. workers
            res1 = requests.post(WORKER_URL, json={"workers": result["workers"]}, timeout=2)
            print(f"[POST] /workers → {res1.status_code}")

            # 2. violations (없으면 건너뜀)
            if result["violations"]:
                res2 = requests.post(VIOLATION_URL, json={"violations": result["violations"]}, timeout=2)
                print(f"[POST] /violations → {res2.status_code}")
            else:
                print("[SKIP] No violations to report.")

            # 3. zones
            res3 = requests.post(ZONE_URL, json={"zones": result["zones"]}, timeout=2)
            print(f"[POST] /zones → {res3.status_code}")

        except requests.RequestException as e:
            print(f"[ERROR] Failed to send: {e}")

        await asyncio.sleep(1.0)

if __name__ == "__main__":
    try:
        asyncio.run(detection_loop())
    except Exception as e:
        print(f"[ERROR] {e}")