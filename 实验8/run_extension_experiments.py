from __future__ import annotations

import csv
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
BEST_WEIGHT = PROJECT_ROOT / "runs" / "train" / "delta_force_soldier" / "weights" / "best.pt"
FRAMES_ROOT = PROJECT_ROOT / "extracted_frames"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

VIDEO_SOURCES = [
    ("video1", FRAMES_ROOT / "video1_segments_fps2"),
    ("video2", FRAMES_ROOT / "video2_segments_fps2"),
    ("video3", FRAMES_ROOT / "video3_segments_fps2"),
    ("video4", FRAMES_ROOT / "video4_segments_fps2"),
]


def image_count(source: Path) -> int:
    return sum(1 for item in source.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES)


def save_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_prediction(name: str, source: Path, save_dir: Path, results) -> dict[str, object]:
    conf_values: list[float] = []
    detected_images = 0
    boxes_total = 0

    for result in results:
        box_count = len(result.boxes)
        if box_count:
            detected_images += 1
            boxes_total += box_count
            conf_values.extend(float(value) for value in result.boxes.conf.cpu().tolist())

    label_dir = save_dir / "labels"
    label_files = len(list(label_dir.glob("*.txt"))) if label_dir.exists() else 0

    avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0.0
    max_conf = max(conf_values) if conf_values else 0.0
    min_conf = min(conf_values) if conf_values else 0.0

    return {
        "experiment": name,
        "source": source.name,
        "input_images": image_count(source),
        "detected_images": detected_images,
        "label_files": label_files,
        "boxes_total": boxes_total,
        "avg_conf": f"{avg_conf:.4f}",
        "min_conf": f"{min_conf:.4f}",
        "max_conf": f"{max_conf:.4f}",
        "save_dir": save_dir.relative_to(PROJECT_ROOT).as_posix(),
    }


def collect_boxes(video_name: str, results) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for result in results:
        image_name = Path(result.path).name
        image_height, image_width = result.orig_shape
        names = result.names

        xyxy = result.boxes.xyxy.cpu().tolist()
        conf = result.boxes.conf.cpu().tolist()
        cls = result.boxes.cls.cpu().tolist()

        for index, (coords, confidence, class_id_float) in enumerate(zip(xyxy, conf, cls), start=1):
            class_id = int(class_id_float)
            rows.append(
                {
                    "video": video_name,
                    "image": image_name,
                    "box_index": index,
                    "x1": f"{coords[0]:.2f}",
                    "y1": f"{coords[1]:.2f}",
                    "x2": f"{coords[2]:.2f}",
                    "y2": f"{coords[3]:.2f}",
                    "conf": f"{float(confidence):.4f}",
                    "cls": class_id,
                    "name": names.get(class_id, str(class_id)),
                    "image_width": image_width,
                    "image_height": image_height,
                }
            )

    return rows


def run_predict(model: YOLO, source: Path, project: Path, name: str, conf: float, iou: float = 0.7):
    return model.predict(
        source=str(source),
        conf=conf,
        iou=iou,
        imgsz=640,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(project),
        name=name,
        exist_ok=True,
    )


def main() -> None:
    if not BEST_WEIGHT.exists():
        raise FileNotFoundError(f"Train the model first. Missing: {BEST_WEIGHT}")

    model = YOLO(str(BEST_WEIGHT))

    multi_video_rows: list[dict[str, object]] = []
    all_box_rows: list[dict[str, object]] = []
    multi_project = PROJECT_ROOT / "runs" / "predict_multi_video"

    for video_name, source in VIDEO_SOURCES:
        results = run_predict(model, source, multi_project, video_name, conf=0.25)
        save_dir = multi_project / video_name
        multi_video_rows.append(summarize_prediction(video_name, source, save_dir, results))
        all_box_rows.extend(collect_boxes(video_name, results))

    conf_rows: list[dict[str, object]] = []
    conf_project = PROJECT_ROOT / "runs" / "predict_conf_compare"
    video4_source = FRAMES_ROOT / "video4_segments_fps2"

    for conf_value, name in [(0.25, "conf025"), (0.60, "conf060")]:
        results = run_predict(model, video4_source, conf_project, name, conf=conf_value)
        conf_rows.append(summarize_prediction(name, video4_source, conf_project / name, results))

    save_csv(
        OUTPUTS_DIR / "multi_video_summary.csv",
        [
            "experiment",
            "source",
            "input_images",
            "detected_images",
            "label_files",
            "boxes_total",
            "avg_conf",
            "min_conf",
            "max_conf",
            "save_dir",
        ],
        multi_video_rows,
    )
    save_csv(
        OUTPUTS_DIR / "conf_compare_summary.csv",
        [
            "experiment",
            "source",
            "input_images",
            "detected_images",
            "label_files",
            "boxes_total",
            "avg_conf",
            "min_conf",
            "max_conf",
            "save_dir",
        ],
        conf_rows,
    )
    save_csv(
        OUTPUTS_DIR / "boxes_info.csv",
        [
            "video",
            "image",
            "box_index",
            "x1",
            "y1",
            "x2",
            "y2",
            "conf",
            "cls",
            "name",
            "image_width",
            "image_height",
        ],
        all_box_rows,
    )

    print("multi_video_summary.csv")
    for row in multi_video_rows:
        print(row)

    print("conf_compare_summary.csv")
    for row in conf_rows:
        print(row)

    print(f"boxes_info.csv rows: {len(all_box_rows)}")


if __name__ == "__main__":
    main()
