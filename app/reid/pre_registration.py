import os
import cv2
import numpy as np
import pickle
import redis
from pathlib import Path
from typing import List, Dict, Optional
from ultralytics import YOLO
import sys
import os
from pathlib import Path

# app 디렉토리 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(current_dir)
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# torchreid 모듈 경로 추가
PROJECT_ROOT = Path(app_dir).parent
TORCHREID_PATH = str(PROJECT_ROOT / "deep-person-reid-master")
if TORCHREID_PATH not in sys.path:
    sys.path.insert(0, TORCHREID_PATH)

from image_processor import ImageProcessor
from config import settings


class PreRegistrationManager:
    """
    사전 등록된 이미지들을 처리하여 특징벡터를 Redis에 저장하는 관리자
    app/pre_img/ 폴더에서 각 global_id별 폴더의 이미지들을 처리
    """
    
    def __init__(self, redis_handler=None):
        # 직접 Redis 연결 (사전 등록 전용)
        self.redis = redis.Redis(
            host=settings.REDIS_CONFIG["host"],
            port=settings.REDIS_CONFIG["port"],
            decode_responses=False
        )
        
        # app/pre_img 폴더 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = os.path.dirname(current_dir)
        self.pre_img_path = Path(app_dir) / "pre_img"
        
        # YOLO 모델 초기화
        model_path = settings.YOLO_MODEL_PATH
        self.yolo_model = YOLO(model_path, task="detect")
        self.class_names = self.yolo_model.names
        
        # ImageProcessor 초기화
        self.image_processor = ImageProcessor()
        
        # 설정
        self.target_width = 640
        self.person_classes = ["person", "saram"]
    
    def register_all_pre_images(self) -> Dict[str, any]:
        """
        app/pre_img/ 폴더의 모든 사전 등록 이미지를 처리
        
        Returns:
            처리 결과 딕셔너리
        """
        if not self.pre_img_path.exists():
            raise FileNotFoundError(f"사전 등록 폴더가 없습니다: {self.pre_img_path}")
        
        print(f"[PreRegistration] 사전 등록 시작: {self.pre_img_path}")
        
        # 모든 global_id 폴더 찾기
        global_id_folders = []
        for item in self.pre_img_path.iterdir():
            if item.is_dir() and item.name.isdigit():
                global_id_folders.append(int(item.name))
        
        if not global_id_folders:
            raise ValueError(f"사전 등록할 global_id 폴더가 없습니다: {self.pre_img_path}")
        
        global_id_folders.sort()
        print(f"[PreRegistration] 발견된 global_id 폴더: {global_id_folders}")
        
        success_count = 0
        failed_count = 0
        failed_global_ids = []
        
        for global_id in global_id_folders:
            try:
                success = self._register_single_global_id(global_id)
                if success:
                    success_count += 1
                    print(f"[PreRegistration] 성공: Global ID {global_id}")
                else:
                    failed_count += 1
                    failed_global_ids.append(global_id)
                    print(f"[PreRegistration] 실패: Global ID {global_id}")
            except Exception as e:
                failed_count += 1
                failed_global_ids.append(global_id)
                print(f"[PreRegistration] 오류 발생 (Global ID {global_id}): {str(e)}")
        
        return {
            "success": failed_count == 0,
            "message": f"사전 등록 완료: {success_count}개 성공, {failed_count}개 실패",
            "success_count": success_count,
            "failed_count": failed_count,
            "failed_global_ids": failed_global_ids
        }
    
    def _register_single_global_id(self, global_id: int) -> bool:
        """
        단일 global_id 폴더의 이미지들을 처리
        
        Args:
            global_id: 처리할 global_id
            
        Returns:
            성공 여부
        """
        folder_path = self.pre_img_path / str(global_id)
        
        if not folder_path.exists():
            raise FileNotFoundError(f"Global ID {global_id} 폴더가 없습니다: {folder_path}")
        
        # 이미지 파일들 찾기 (10장 필요)
        image_files = self._find_image_files(folder_path)
        
        if len(image_files) < 10:
            raise ValueError(f"Global ID {global_id}: 이미지가 10장 미만입니다. (발견: {len(image_files)}장)")
        
        if len(image_files) > 10:
            print(f"[PreRegistration] 경고: Global ID {global_id}에서 10장 초과 이미지 발견. 처음 10장만 사용합니다.")
            image_files = image_files[:10]
        
        # 각 이미지에서 사람 탐지 및 feature 추출
        features = []
        for i, image_file in enumerate(image_files):
            try:
                feature = self._process_single_image(image_file, global_id, i + 1)
                if feature is not None:
                    features.append(feature)
                else:
                    raise ValueError(f"이미지 {image_file.name}에서 feature 추출 실패")
            except Exception as e:
                raise ValueError(f"이미지 {image_file.name} 처리 중 오류: {str(e)}")
        
        if len(features) != 10:
            raise ValueError(f"Global ID {global_id}: 10개의 feature를 추출하지 못했습니다. (추출: {len(features)}개)")
        
        # Redis에 저장
        self._store_features_to_redis(global_id, features)
        
        print(f"[PreRegistration] Global ID {global_id}: 10개 feature 저장 완료")
        return True
    
    def _find_image_files(self, folder_path: Path) -> List[Path]:
        """
        폴더에서 이미지 파일들을 찾기
        
        Args:
            folder_path: 이미지 폴더 경로
            
        Returns:
            이미지 파일 경로 리스트
        """
        # 지원하는 이미지 확장자
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.TIF'}
        
        image_files = []
        for ext in image_extensions:
            image_files.extend(folder_path.glob(f"*{ext}"))
        
        # 파일명으로 정렬 (10_1, 10_2, ... 순서)
        image_files.sort(key=lambda x: x.name)
        
        return image_files
    
    def _process_single_image(self, image_path: Path, global_id: int, image_index: int) -> Optional[np.ndarray]:
        """
        단일 이미지 처리: 사람 탐지 → 크롭 → feature 추출
        
        Args:
            image_path: 이미지 파일 경로
            global_id: global_id
            image_index: 이미지 인덱스 (1~10)
            
        Returns:
            추출된 feature 벡터 또는 None
        """
        # 이미지 로드
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        
        # 사람 탐지
        detections = self.detect_person_only(image)
        
        if not detections:
            raise ValueError(f"이미지에서 사람을 탐지할 수 없습니다: {image_path}")
        
        # 가장 높은 confidence를 가진 탐지 결과 사용
        best_detection = max(detections, key=lambda x: x[4])  # confidence 기준
        x1, y1, x2, y2, conf = best_detection
        
        # 바운딩 박스 좌표를 정수로 변환
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # 이미지 크롭
        crop = image[y1:y2, x1:x2]
        crop = cv2.resize(crop, (128, 256))
        
        if crop.size == 0:
            raise ValueError(f"크롭된 이미지가 비어있습니다: {image_path}")
        
        # Feature 추출
        feature = self.image_processor.extract_feature(crop)
        
        if feature is None or len(feature) == 0:
            raise ValueError(f"Feature 추출에 실패했습니다: {image_path}")
        
        print(f"[PreRegistration] Global ID {global_id}, 이미지 {image_index}: 크롭 크기 {crop.shape}, Feature 차원 {len(feature)}")
        
        return feature
    
    def detect_person_only(self, image: np.ndarray) -> List[List[float]]:
        """
        이미지에서 사람만 탐지 (YOLO만 사용)
        사전 등록용으로 사용
        
        Args:
            image: 입력 이미지
            
        Returns:
            detections: 사람 탐지 결과 [x1, y1, x2, y2, confidence]
        """
        # 원본 이미지 크기 저장
        original_height, original_width = image.shape[:2]
        
        # 이미지 리사이즈
        scale = self.target_width / original_width
        target_height = int(original_height * scale)
        resized_image = cv2.resize(image, (self.target_width, target_height))
        
        # YOLO 탐지 수행
        results = self.yolo_model(resized_image, verbose=False)[0]
        
        # 탐지 결과 변환
        dets = []
        boxes = results.boxes
        
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # 사람 클래스만 필터링
                class_name = self.class_names[cls_id].lower()
                if class_name in self.person_classes:
                    # xyxy 형식으로 변환
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    dets.append([x1, y1, x2, y2, conf])
        
        # 사람 클래스만 필터링
        person_detections = []
        for det in dets:
            x1, y1, x2, y2, conf = det
            # confidence threshold 적용
            if conf >= 0.5:
                person_detections.append([x1, y1, x2, y2, conf])
        
        # 좌표를 원본 이미지 크기로 변환
        scale_x = original_width / self.target_width
        scale_y = original_height / target_height
        
        for det in person_detections:
            det[0] *= scale_x  # x1
            det[1] *= scale_y  # y1
            det[2] *= scale_x  # x2
            det[3] *= scale_y  # y2
        
        return person_detections
    
    def _store_features_to_redis(self, global_id: int, features: List[np.ndarray]):
        """
        추출된 features를 Redis에 저장
        
        Args:
            global_id: global_id
            features: feature 벡터 리스트 (10개)
        """
        # Redis 키 생성
        redis_key = f"global_track_pre:{global_id}"
        
        # 저장할 데이터 구조 (last_seen, last_bbox 제외)
        track_data = {
            'features': features
        }
        
        # Redis에 저장 (TTL 무제한)
        serialized_data = pickle.dumps(track_data)
        self.redis.set(redis_key, serialized_data)
        
        print(f"[PreRegistration] Redis 저장 완료: {redis_key}, Feature 개수: {len(features)}")
    
    def get_pre_registered_features(self, global_id: int) -> Optional[List[np.ndarray]]:
        """
        사전 등록된 features 조회
        
        Args:
            global_id: 조회할 global_id
            
        Returns:
            feature 벡터 리스트 또는 None
        """
        redis_key = f"global_track_pre:{global_id}"
        data = self.redis.get(redis_key)
        
        if data is None:
            return None
        
        try:
            track_data = pickle.loads(data)
            return track_data.get('features', [])
        except Exception as e:
            print(f"[PreRegistration] Redis 데이터 역직렬화 실패: {str(e)}")
            return None
    
    def list_all_pre_registered(self) -> List[int]:
        """
        사전 등록된 모든 global_id 목록 조회
        
        Returns:
            global_id 리스트
        """
        keys = self.redis.keys("global_track_pre:*")
        global_ids = []
        
        for key in keys:
            try:
                key_str = key.decode()
                global_id = int(key_str.split(":")[1])
                global_ids.append(global_id)
            except Exception as e:
                print(f"[PreRegistration] 키 파싱 오류: {key}, {str(e)}")
                continue
        
        return sorted(global_ids)
    
    def clear_pre_registered_data(self, global_id: Optional[int] = None):
        """
        사전 등록 데이터 삭제
        
        Args:
            global_id: 삭제할 global_id (None이면 모든 데이터 삭제)
        """
        if global_id is not None:
            redis_key = f"global_track_pre:{global_id}"
            self.redis.delete(redis_key)
            print(f"[PreRegistration] Global ID {global_id} 사전 등록 데이터 삭제 완료")
        else:
            keys = self.redis.keys("global_track_pre:*")
            if keys:
                self.redis.delete(*keys)
                print(f"[PreRegistration] 모든 사전 등록 데이터 삭제 완료 ({len(keys)}개)")


def main():
    """사전 등록 실행 함수 (테스트용)"""
    try:
        # PreRegistrationManager 초기화
        pre_reg_manager = PreRegistrationManager()
        
        # 모든 사전 등록 이미지 처리
        result = pre_reg_manager.register_all_pre_images()
        
        if result["success"]:
            print(f"✅ 사전 등록 성공: {result['message']}")
            
            # 등록된 데이터 확인
            registered_ids = pre_reg_manager.list_all_pre_registered()
            print(f"📋 등록된 Global ID 목록: {registered_ids}")
            
            # 각 Global ID별 feature 개수 확인
            for global_id in registered_ids:
                features = pre_reg_manager.get_pre_registered_features(global_id)
                if features:
                    print(f"🔍 Global ID {global_id}: {len(features)}개 feature")
                else:
                    print(f"❌ Global ID {global_id}: feature 조회 실패")
        else:
            print(f"❌ 사전 등록 실패: {result['message']}")
            if result['failed_global_ids']:
                print(f"실패한 Global ID: {result['failed_global_ids']}")
                
    except Exception as e:
        print(f"💥 사전 등록 중 오류 발생: {str(e)}")


if __name__ == "__main__":
    main()
