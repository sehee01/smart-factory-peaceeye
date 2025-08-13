#!/usr/bin/env python3
"""
사전 등록 기능 테스트 스크립트
"""

import sys
import argparse
from pathlib import Path

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torchreid.utils.feature_extractor import FeatureExtractor
from reid.redis_handler import FeatureStoreRedisHandler
from reid.pre_registration import create_pre_registration_manager
from config import settings


def main():
    parser = argparse.ArgumentParser(description="사전 등록 기능 테스트")
    parser.add_argument(
        '--folder',
        type=str,
        required=True,
        help='등록할 이미지들이 있는 폴더 경로'
    )
    parser.add_argument(
        '--global_id',
        type=int,
        required=True,
        help='등록할 글로벌 ID'
    )
    parser.add_argument(
        '--camera_id',
        type=str,
        default='pre_registered',
        help='카메라 ID (기본값: pre_registered)'
    )
    parser.add_argument(
        '--redis_host',
        type=str,
        default='localhost',
        help='Redis 서버 호스트'
    )
    parser.add_argument(
        '--redis_port',
        type=int,
        default=6379,
        help='Redis 서버 포트'
    )
    parser.add_argument(
        '--action',
        type=str,
        choices=['register', 'list', 'remove', 'clear'],
        default='register',
        help='수행할 작업 (기본값: register)'
    )
    
    args = parser.parse_args()
    
    # Redis 핸들러 초기화
    redis_handler = FeatureStoreRedisHandler(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        feature_ttl=settings.REID_CONFIG.get("ttl", 300)
    )
    
    # Feature Extractor 초기화
    device = settings.FEATURE_EXTRACTOR_CONFIG["device"]
    if device == "auto":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    feature_extractor = FeatureExtractor(
        model_name=settings.FEATURE_EXTRACTOR_CONFIG["model_name"],
        model_path=settings.FEATURE_EXTRACTOR_CONFIG["model_path"],
        device=device
    )
    
    # PreRegistrationManager 생성
    pre_reg_manager = create_pre_registration_manager(redis_handler, feature_extractor)
    
    print(f"[PreRegistration Test] 작업: {args.action}")
    print(f"[PreRegistration Test] Redis: {args.redis_host}:{args.redis_port}")
    
    if args.action == 'register':
        # 폴더에서 이미지 등록
        print(f"[PreRegistration Test] 폴더 등록: {args.folder}")
        print(f"[PreRegistration Test] Global ID: {args.global_id}")
        print(f"[PreRegistration Test] Camera ID: {args.camera_id}")
        
        result = pre_reg_manager.register_images_from_folder(
            folder_path=args.folder,
            global_id=args.global_id,
            camera_id=args.camera_id
        )
        
        print(f"\n[PreRegistration Test] 결과:")
        print(f"  성공: {result['success']}")
        print(f"  메시지: {result['message']}")
        print(f"  처리된 파일 수: {result['processed_count']}")
        print(f"  실패한 파일 수: {result['failed_count']}")
        
        if result['failed_files']:
            print(f"  실패한 파일들:")
            for failed_file in result['failed_files']:
                print(f"    - {failed_file}")
    
    elif args.action == 'list':
        # 등록된 특징 벡터 목록 조회
        print(f"[PreRegistration Test] 등록된 특징 벡터 목록 조회")
        
        features = pre_reg_manager.list_registered_features(camera_id=args.camera_id)
        
        if features:
            print(f"\n[PreRegistration Test] 등록된 특징 벡터 ({len(features)}개):")
            for feature in features:
                print(f"  Global ID: {feature['global_id']}")
                print(f"  Camera ID: {feature['camera_id']}")
                print(f"  Local Track ID: {feature['local_track_id']}")
                print(f"  Features Count: {feature['features_count']}")
                print(f"  Last Seen: {feature['last_seen']}")
                print(f"  BBox: {feature['bbox']}")
                print("  ---")
        else:
            print(f"\n[PreRegistration Test] 등록된 특징 벡터가 없습니다.")
    
    elif args.action == 'remove':
        # 특정 글로벌 ID의 특징 벡터 삭제
        print(f"[PreRegistration Test] Global ID {args.global_id} 삭제")
        
        success = pre_reg_manager.remove_registered_features(
            global_id=args.global_id,
            camera_id=args.camera_id
        )
        
        if success:
            print(f"[PreRegistration Test] 삭제 성공")
        else:
            print(f"[PreRegistration Test] 삭제 실패")
    
    elif args.action == 'clear':
        # 모든 사전 등록된 특징 벡터 삭제
        print(f"[PreRegistration Test] 모든 사전 등록 데이터 삭제")
        
        success = pre_reg_manager.clear_all_pre_registered(camera_id=args.camera_id)
        
        if success:
            print(f"[PreRegistration Test] 모든 데이터 삭제 성공")
        else:
            print(f"[PreRegistration Test] 데이터 삭제 실패")
    
    print(f"\n[PreRegistration Test] 작업 완료")


if __name__ == '__main__':
    main()
