# ai/run_yolo_test.py
"""
YOLOv8 sanity-check script
──────────────────────────
• 기본 가중치 경로 : ai/weights/best.pt
• 단일 이미지 / 폴더 / 동영상 / 웹캠 스트림을 모두 처리
• --save 옵션을 주면 ai/output/ 에 바운딩박스가 그려진 결과 저장
"""

from pathlib import Path
import argparse, sys, subprocess, json
import torch
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]          # smartfactoryPeaceeye/
DEFAULT_WEIGHTS = ROOT / "weights" / "best.pt"
DEFAULT_OUTDIR  = ROOT / "ai" / "output"
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True,
                   help="이미지/폴더/동영상 경로 또는 웹캠 인덱스(0,1,...)")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                   help=f"YOLO .pt (default: {DEFAULT_WEIGHTS})")
    p.add_argument("--device", default="cuda",
                   help="'cuda', GPU 번호(0|1|...), or 'cpu'")
    p.add_argument("--save", action="store_true",
                   help="결과를 ai/output/에 저장")
    return p.parse_args()

def check_cuda(device_flag: str):
    if device_flag != "cpu" and not torch.cuda.is_available():
        sys.exit("❌ GPU(CUDA)를 찾을 수 없습니다. --device cpu 로 실행하세요.")

def main():
    args = parse_args()
    check_cuda(args.device)

    print(f"🟢 Loading model : {args.weights}")
    print(f"🟢 Device        : {args.device}")

    model = YOLO(str(args.weights))
    model.to(args.device)

    outdir = None
    if args.save:
        outdir = model.predict(
            args.source,
            save=True,
            project=str(DEFAULT_OUTDIR),
            name="run"
        )[0].save_dir
    else:
        model.predict(args.source, save=False)

    print("✅ Inference finished.")

    if outdir:
        print(f"📂 Saved to : {outdir}")

if __name__ == "__main__":
    main()
