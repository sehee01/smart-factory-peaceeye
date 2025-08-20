# ultralytics_tracker.py
from ultralytics import YOLO
import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Union
from pathlib import Path
import importlib


def _has_valid_tensorrt() -> bool:
    """TensorRT 모듈이 정상 설치되어 있고 __version__을 노출하는지 확인"""
    try:
        trt = importlib.import_module("tensorrt")
        return hasattr(trt, "__version__")
    except Exception:
        return False


def _guess_pt_fallback(engine_path: Union[str, Path]) -> Union[str, None]:
    """
    .engine/.trt 경로가 들어오면 같은 이름의 .pt를 우선 찾고,
    settings.YOLO_MODEL_PATH 같은 기본 경로도 시도할 수 있도록 훅을 남김.
    """
    p = Path(str(engine_path))
    pt_candidate = p.with_suffix(".pt")
    if pt_candidate.exists():
        return str(pt_candidate)
    # 필요시 다른 규칙 추가 가능
    return None


class UltralyticsTrackerManager:
    """
    Ultralytics YOLO의 내장 tracking 기능을 사용하는 매니저
    ByteTrack 대신 Ultralytics의 내장 tracker 사용
    """

    def __init__(self, model_path: Union[str, YOLO], tracker_config: dict):
        self.tracker_config = tracker_config or {}
        self.target_width = self.tracker_config.get("target_width", 640)
        self.person_classes = ["person", "saram"]  # 탐지 대상 클래스
        self.original_frame_shape = None
        self.track_history = {}
        self.max_history_length = self.tracker_config.get("track_buffer", 30)

        # 추론 파라미터
        self.conf = float(self.tracker_config.get("track_thresh", 0.5))  # detection conf
        self.iou = float(self.tracker_config.get("match_thresh", 0.8))   # NMS IOU
        self.tracker_yaml = self.tracker_config.get("tracker_yaml", "bytetrack.yaml")

        # [PATCH] 모델 로드: TRT 엔진이면 TRT 유효성 점검 후 없으면 .pt로 폴백
        self._pt_fallback_used = False
        self.model = self._load_model_with_fallback(model_path)

        # class_names 안전 접근
        names = None
        if hasattr(self.model, "names"):
            names = self.model.names
        elif hasattr(self.model, "model") and hasattr(self.model.model, "names"):
            names = self.model.model.names

        if isinstance(names, list):
            self.class_names = {i: n for i, n in enumerate(names)}
        elif isinstance(names, dict):
            self.class_names = names
        else:
            self.class_names = {0: "person"}  # 폴백

    # [PATCH] 안전 로더
    def _load_model_with_fallback(self, mp: Union[str, YOLO]):
        if isinstance(mp, YOLO):
            return mp

        path_str = str(mp)
        suffix = Path(path_str).suffix.lower()
        is_engine = suffix in (".engine", ".trt")

        if is_engine and not _has_valid_tensorrt():
            # TensorRT가 불완전 → .pt로 폴백 시도
            pt = _guess_pt_fallback(path_str)
            if pt:
                print(f"[YOLO] TensorRT unavailable → fallback to PT weights: {pt}")
                self._pt_fallback_used = True
                return YOLO(pt)
            else:
                # 폴백 불가 시 안내 후 예외
                raise RuntimeError(
                    "TensorRT engine provided but TensorRT is not properly installed "
                    "and no .pt fallback was found. Install correct TensorRT or provide a .pt model."
                )
        # 일반적인 경우: 그대로 로드
        return YOLO(path_str)

    def detect_and_track(self, frame, frame_id: int):
        """프레임에서 객체 탐지 및 추적 수행 (Ultralytics 내장 tracking 사용)"""
        self.original_frame_shape = frame.shape[:2]  # (H, W)
        resized_frame = self._resize_frame(frame)

        # [PATCH] TRT 오류 발생 시 1회 PT 재시도
        try:
            results = self._run_tracking(resized_frame)
        except AttributeError as e:
            msg = str(e)
            if "tensorrt" in msg.lower() or "__version__" in msg.lower():
                # 런타임 시도 중에 TRT가 터졌음 → .pt로 재로딩 후 재시도
                if not self._pt_fallback_used:
                    pt = _guess_pt_fallback(getattr(self.model, "ckpt_path", "model.engine"))
                    if pt:
                        print(f"[YOLO] Caught TRT error at runtime → reload PT weights: {pt}")
                        self.model = YOLO(pt)
                        self._pt_fallback_used = True
                        results = self._run_tracking(resized_frame)
                    else:
                        raise
                else:
                    raise
            else:
                raise

        track_list = self._convert_tracking_results(results, frame_id)
        track_list = self._convert_coordinates_to_original(track_list)
        return track_list

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """프레임을 목표 너비에 맞춰 비율 유지 리사이즈"""
        h, w = frame.shape[:2]
        target_w = int(self.target_width)
        if target_w <= 0 or target_w == w:
            return frame
        scale = target_w / float(w)
        target_h = int(round(h * scale))
        return cv2.resize(frame, (target_w, target_h))

    def _run_tracking(self, frame: np.ndarray) -> Any:
        """YOLO 모델로 객체 탐지 및 추적 수행 (Ultralytics track API)"""
        use_half = bool(torch.cuda.is_available())  # CPU면 half 비활성

        results_list = self.model.track(
            source=frame,
            verbose=False,
            half=use_half,
            persist=True,                 # 프레임 간 추적 상태 유지
            tracker=self.tracker_yaml,    # Ultralytics ByteTrack 설정
            conf=self.conf,               # detection confidence threshold
            iou=self.iou                  # NMS IOU threshold
        )
        return results_list[0] if isinstance(results_list, (list, tuple)) else results_list

    def _convert_tracking_results(self, results: Any, frame_id: int) -> List[Dict]:
        """Ultralytics tracking 결과를 표준 형식으로 변환"""
        track_list: List[Dict] = []

        boxes = getattr(results, "boxes", None)
        if boxes is None:
            return track_list

        ids = getattr(boxes, "id", None)
        xyxy = getattr(boxes, "xyxy", None)
        conf = getattr(boxes, "conf", None)
        cls  = getattr(boxes, "cls", None)

        if ids is None or xyxy is None:
            return track_list

        # 텐서 → 넘파이
        def to_np(t, default=None):
            if t is None:
                return default
            return t.detach().cpu().numpy() if hasattr(t, "detach") else np.array(t)

        ids_np  = to_np(ids, default=np.array([], dtype=int))
        xyxy_np = to_np(xyxy, default=np.zeros((0, 4), dtype=float))
        conf_np = to_np(conf, default=np.zeros(len(xyxy_np), dtype=float))
        cls_np  = to_np(cls,  default=np.zeros(len(xyxy_np), dtype=int))

        def get_class_name(cid: int) -> str:
            if isinstance(self.class_names, dict):
                return str(self.class_names.get(int(cid), str(cid)))
            elif isinstance(self.class_names, list) and 0 <= int(cid) < len(self.class_names):
                return str(self.class_names[int(cid)])
            return str(cid)

        person_aliases = {s.lower() for s in self.person_classes}

        for i in range(len(xyxy_np)):
            track_id = int(ids_np[i]) if len(ids_np) > i else -1
            x1, y1, x2, y2 = [float(v) for v in xyxy_np[i].tolist()]
            c  = float(conf_np[i]) if len(conf_np) > i else 0.0
            cid = int(cls_np[i]) if len(cls_np) > i else 0
            cname = get_class_name(cid)

            if cname.lower() in person_aliases:
                track_list.append({
                    "track_id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": c,
                    "class_id": cid,
                    "class_name": cname
                })
                self._update_track_history(track_id, [x1, y1, x2, y2], frame_id)

        return track_list

    def _update_track_history(self, track_id: int, bbox: List[float], frame_id: int):
        """추적 히스토리 업데이트"""
        if track_id not in self.track_history:
            self.track_history[track_id] = []

        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        self.track_history[track_id].append({
            "frame_id": frame_id,
            "center": [center_x, center_y],
            "bbox": bbox
        })

        if len(self.track_history[track_id]) > self.max_history_length:
            self.track_history[track_id] = self.track_history[track_id][-self.max_history_length:]

    def _convert_coordinates_to_original(self, track_list: List[Dict]) -> List[Dict]:
        """리사이즈 좌표 → 원본 프레임 좌표로 변환"""
        if self.original_frame_shape is None:
            return track_list

        original_h, original_w = self.original_frame_shape
        target_w = float(self.target_width)
        if target_w <= 0:
            return track_list

        scale = original_w / target_w
        for track in track_list:
            x1, y1, x2, y2 = track["bbox"]
            track["bbox"] = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]
        return track_list

    def get_track_history(self, track_id: int) -> List[Dict]:
        """특정 track_id의 히스토리 반환"""
        return self.track_history.get(track_id, [])

    def clear_track_history(self):
        """모든 추적 히스토리 초기화"""
        self.track_history.clear()
