import os
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from torchreid.utils.feature_extractor import FeatureExtractor
from .redis_handler import FeatureStoreRedisHandler
from config import settings


class PreRegistrationManager:
    """
    사전 등록된 이미지들을 처리하여 특징벡터를 Redis에 저장하는 관리자
    """
    
    def __init__(self, redis_handler: FeatureStoreRedisHandler, feature_extractor: FeatureExtractor):
        self.redis_handler = redis_handler
        self.feature_extractor = feature_extractor
        
    def register_images_from_folder(self, folder_path: str, global_id: int, camera_id: str = "pre_registered") -> Dict[str, any]:
        """
        폴더 내의 모든 이미지를 처리하여 Redis에 저장
        
        Args:
            folder_path: 이미지가 저장된 폴더 경로
            global_id: 등록할 글로벌 ID
            camera_id: 카메라 ID (기본값: "pre_registered")
            
        Returns:
            처리 결과 딕셔너리
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {folder_path}")
        
        # 지원하는 이미지 확장자
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 폴더 내 모든 이미지 파일 찾기
        image_files = []
        for ext in image_extensions:
            image_files.extend(folder_path.glob(f"*{ext}"))
            image_files.extend(folder_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            return {
                "success": False,
                "message": f"폴더에 이미지 파일이 없습니다: {folder_path}",
                "processed_count": 0,
                "failed_count": 0
            }
        
        print(f"[PreRegistration] {len(image_files)}개의 이미지 파일을 발견했습니다.")
        
        processed_count = 0
        failed_count = 0
        failed_files = []
        
        for image_file in image_files:
            try:
                success = self._process_single_image(
                    image_path=str(image_file),
                    global_id=global_id,
                    camera_id=camera_id,
                    local_track_id=processed_count + 1
                )
                
                if success:
                    processed_count += 1
                    print(f"[PreRegistration] 성공: {image_file.name}")
                else:
                    failed_count += 1
                    failed_files.append(image_file.name)
                    print(f"[PreRegistration] 실패: {image_file.name}")
                    
            except Exception as e:
                failed_count += 1
                failed_files.append(image_file.name)
                print(f"[PreRegistration] 오류 발생 ({image_file.name}): {str(e)}")
        
        return {
            "success": processed_count > 0,
            "message": f"처리 완료: {processed_count}개 성공, {failed_count}개 실패",
            "processed_count": processed_count,
            "failed_count": failed_count,
            "failed_files": failed_files
        }
    
    def register_single_image(self, image_path: str, global_id: int, camera_id: str = "pre_registered", 
                            local_track_id: int = 1) -> bool:
        """
        단일 이미지를 처리하여 Redis에 저장
        
        Args:
            image_path: 이미지 파일 경로
            global_id: 등록할 글로벌 ID
            camera_id: 카메라 ID
            local_track_id: 로컬 트랙 ID
            
        Returns:
            성공 여부
        """
        return self._process_single_image(image_path, global_id, camera_id, local_track_id)
    
    def _process_single_image(self, image_path: str, global_id: int, camera_id: str, 
                            local_track_id: int) -> bool:
        """
        단일 이미지 처리 내부 메서드
        
        Args:
            image_path: 이미지 파일 경로
            global_id: 글로벌 ID
            camera_id: 카메라 ID
            local_track_id: 로컬 트랙 ID
            
        Returns:
            성공 여부
        """
        try:
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                print(f"[PreRegistration] 이미지를 로드할 수 없습니다: {image_path}")
                return False
            
            # 이미지 크기 확인
            height, width = image.shape[:2]
            if height == 0 or width == 0:
                print(f"[PreRegistration] 이미지 크기가 유효하지 않습니다: {image_path}")
                return False
            
            # 전체 이미지를 하나의 객체로 간주하여 처리
            bbox = [0, 0, width, height]  # 전체 이미지 영역
            
            # 특징 벡터 추출
            feature = self._extract_feature_from_image(image)
            if feature is None:
                print(f"[PreRegistration] 특징 벡터 추출 실패: {image_path}")
                return False
            
            # Redis에 저장 (메타데이터와 함께)
            self.redis_handler.store_feature_with_metadata(
                global_id=global_id,
                camera_id=camera_id,
                local_track_id=local_track_id,
                feature=feature,
                bbox=bbox,
                frame_id=0  # 사전 등록은 프레임 0으로 설정
            )
            
            print(f"[PreRegistration] 저장 완료: Global ID {global_id}, Camera {camera_id}, Local {local_track_id}")
            return True
            
        except Exception as e:
            print(f"[PreRegistration] 처리 중 오류 발생: {str(e)}")
            return False
    
    def _extract_feature_from_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        이미지에서 특징 벡터 추출
        
        Args:
            image: 입력 이미지
            
        Returns:
            특징 벡터 또는 None
        """
        try:
            if self.feature_extractor is not None:
                # 전문적인 feature extractor 사용
                return self._extract_feature_with_extractor(image)
            else:
                # 단순한 RGB 평균 feature (fallback)
                return self._extract_feature_simple(image)
                
        except Exception as e:
            print(f"[PreRegistration] 특징 벡터 추출 오류: {str(e)}")
            return None
    
    def _extract_feature_with_extractor(self, image: np.ndarray) -> np.ndarray:
        """
        전문적인 feature extractor를 사용한 특징 벡터 추출
        """
        if image.size == 0:
            return np.zeros(512)  # osnet_ibn_x1_0의 feature dimension
        
        # 패딩 없이 단순 resize
        target_size = settings.FEATURE_EXTRACTOR_CONFIG["target_size"]
        resized_image = cv2.resize(image, target_size)
        normalized_image = resized_image.astype(np.float32) / 255.0
        
        # Feature 추출
        with torch.no_grad():
            feature = self.feature_extractor([normalized_image]).cpu().numpy()
            return feature.flatten()
    
    def _extract_feature_simple(self, image: np.ndarray) -> np.ndarray:
        """
        단순한 RGB 평균 특징 벡터 추출 (fallback)
        """
        if image.size == 0:
            return np.zeros(3)
        feature = image.mean(axis=(0, 1))  # RGB 평균
        return feature / 255.0
    
    def list_registered_features(self, camera_id: str = "pre_registered") -> List[Dict]:
        """
        사전 등록된 특징 벡터 목록 조회
        
        Args:
            camera_id: 카메라 ID
            
        Returns:
            등록된 특징 벡터 정보 리스트
        """
        try:
            # Redis에서 해당 카메라의 모든 키 조회
            keys = self.redis_handler.redis.keys(f"global_track_data:*:{camera_id}:*")
            
            registered_features = []
            for key in keys:
                try:
                    # 키에서 정보 추출
                    key_parts = key.decode().split(":")
                    if len(key_parts) >= 4:
                        global_id = int(key_parts[1])
                        local_track_id = int(key_parts[3])
                        
                        # 데이터 로드
                        data = self.redis_handler.redis.get(key)
                        if data:
                            track_info = self.redis_handler._deserialize_data(data)
                            if isinstance(track_info, dict):
                                registered_features.append({
                                    "global_id": global_id,
                                    "camera_id": camera_id,
                                    "local_track_id": local_track_id,
                                    "features_count": len(track_info.get('features', [])),
                                    "last_seen": track_info.get('last_seen', 0),
                                    "bbox": track_info.get('last_bbox', [0, 0, 0, 0])
                                })
                except Exception as e:
                    print(f"[PreRegistration] 키 처리 중 오류: {key}, {str(e)}")
                    continue
            
            return registered_features
            
        except Exception as e:
            print(f"[PreRegistration] 등록된 특징 벡터 조회 중 오류: {str(e)}")
            return []
    
    def remove_registered_features(self, global_id: int, camera_id: str = "pre_registered") -> bool:
        """
        사전 등록된 특징 벡터 삭제
        
        Args:
            global_id: 삭제할 글로벌 ID
            camera_id: 카메라 ID
            
        Returns:
            성공 여부
        """
        try:
            # 해당 글로벌 ID의 모든 키 찾기
            keys = self.redis_handler.redis.keys(f"global_track_data:{global_id}:{camera_id}:*")
            
            if not keys:
                print(f"[PreRegistration] 삭제할 데이터가 없습니다: Global ID {global_id}")
                return False
            
            # 키 삭제
            deleted_count = self.redis_handler.redis.delete(*keys)
            print(f"[PreRegistration] {deleted_count}개의 키가 삭제되었습니다: Global ID {global_id}")
            
            return deleted_count > 0
            
        except Exception as e:
            print(f"[PreRegistration] 특징 벡터 삭제 중 오류: {str(e)}")
            return False
    
    def clear_all_pre_registered(self, camera_id: str = "pre_registered") -> bool:
        """
        모든 사전 등록된 특징 벡터 삭제
        
        Args:
            camera_id: 카메라 ID
            
        Returns:
            성공 여부
        """
        try:
            # 해당 카메라의 모든 키 찾기
            keys = self.redis_handler.redis.keys(f"global_track_data:*:{camera_id}:*")
            
            if not keys:
                print(f"[PreRegistration] 삭제할 사전 등록 데이터가 없습니다: Camera {camera_id}")
                return True
            
            # 키 삭제
            deleted_count = self.redis_handler.redis.delete(*keys)
            print(f"[PreRegistration] {deleted_count}개의 사전 등록 키가 삭제되었습니다: Camera {camera_id}")
            
            return True
            
        except Exception as e:
            print(f"[PreRegistration] 모든 사전 등록 데이터 삭제 중 오류: {str(e)}")
            return False


def create_pre_registration_manager(redis_handler: FeatureStoreRedisHandler, 
                                  feature_extractor: FeatureExtractor) -> PreRegistrationManager:
    """
    PreRegistrationManager 인스턴스 생성 헬퍼 함수
    
    Args:
        redis_handler: Redis 핸들러
        feature_extractor: 특징 벡터 추출기
        
    Returns:
        PreRegistrationManager 인스턴스
    """
    return PreRegistrationManager(redis_handler, feature_extractor)
