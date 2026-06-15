from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "my_dataset" / "delta_force.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "train"


def main() -> None:
    model = YOLO("yolov8n.pt")
    model.train(
        task="detect",
        data=str(DATA_YAML),
        epochs=50,
        batch=8,
        imgsz=640,
        workers=1,
        device=0,
        project=str(RUNS_DIR),
        name="delta_force_soldier",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
