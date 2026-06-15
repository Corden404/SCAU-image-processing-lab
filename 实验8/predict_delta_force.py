from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
BEST_WEIGHT = PROJECT_ROOT / "runs" / "train" / "delta_force_soldier" / "weights" / "best.pt"
SOURCE = PROJECT_ROOT / "extracted_frames" / "video4_segments_fps2"
PREDICT_DIR = PROJECT_ROOT / "runs" / "predict"


def main() -> None:
    if not BEST_WEIGHT.exists():
        raise FileNotFoundError(f"Train the model first. Missing: {BEST_WEIGHT}")

    model = YOLO(str(BEST_WEIGHT))
    model.predict(
        source=str(SOURCE),
        conf=0.25,
        imgsz=640,
        save=True,
        project=str(PREDICT_DIR),
        name="delta_force_soldier",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
