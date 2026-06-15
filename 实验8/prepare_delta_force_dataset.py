from __future__ import annotations

import argparse
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LABEL_ZIP = PROJECT_ROOT / "labels_my-project-name_2026-06-13-08-24-48.zip"
FRAMES_DIR = PROJECT_ROOT / "extracted_frames"
DATASET_DIR = PROJECT_ROOT / "my_dataset"
CLASS_NAMES = ["soldier"]
TRAIN_RATIO = 0.8
SEED = 20260613


@dataclass
class LabelStats:
    image_count: int
    label_count: int
    missing_labels: list[str]
    extra_labels: list[str]
    empty_label_count: int
    object_count: int


def iter_images(frames_dir: Path) -> list[Path]:
    return sorted(frames_dir.rglob("*.jpg"))


def read_labels(label_zip: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with zipfile.ZipFile(label_zip) as archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".txt"):
                continue
            name = Path(info.filename).name
            labels[name] = archive.read(info).decode("utf-8-sig").strip()
    return labels


def validate_yolo_line(line: str, label_name: str, line_no: int) -> int:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"{label_name}:{line_no} should have 5 fields, got {len(parts)}")

    class_id = int(parts[0])
    if class_id < 0 or class_id >= len(CLASS_NAMES):
        raise ValueError(f"{label_name}:{line_no} class_id {class_id} is outside configured classes")

    values = [float(value) for value in parts[1:]]
    x_center, y_center, width, height = values
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"{label_name}:{line_no} normalized values must be in [0, 1]")
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"{label_name}:{line_no} width and height must be positive")
    return 1


def validate(images: list[Path], labels: dict[str, str]) -> LabelStats:
    image_stems = {image.stem for image in images}
    label_stems = {Path(name).stem for name in labels}
    missing_labels = sorted(image_stems - label_stems)
    extra_labels = sorted(label_stems - image_stems)

    object_count = 0
    empty_label_count = 0
    for label_name, content in labels.items():
        if not content.strip():
            empty_label_count += 1
            continue
        for line_no, line in enumerate(content.splitlines(), start=1):
            if line.strip():
                object_count += validate_yolo_line(line, label_name, line_no)

    return LabelStats(
        image_count=len(images),
        label_count=len(labels),
        missing_labels=missing_labels,
        extra_labels=extra_labels,
        empty_label_count=empty_label_count,
        object_count=object_count,
    )


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def copy_dataset(images: list[Path], labels: dict[str, str]) -> None:
    rng = random.Random(SEED)
    shuffled = images[:]
    rng.shuffle(shuffled)

    split_at = round(len(shuffled) * TRAIN_RATIO)
    splits = {
        "train": sorted(shuffled[:split_at]),
        "val": sorted(shuffled[split_at:]),
    }

    for split, split_images in splits.items():
        for image_path in split_images:
            target_image = DATASET_DIR / "images" / split / image_path.name
            target_label = DATASET_DIR / "labels" / split / f"{image_path.stem}.txt"
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, target_image)
            write_text(target_label, labels.get(f"{image_path.stem}.txt", ""))

    yaml_content = "\n".join(
        [
            "path: .",
            "train: images/train",
            "val: images/val",
            "",
            f"nc: {len(CLASS_NAMES)}",
            f"names: [{', '.join(CLASS_NAMES)}]",
            "",
        ]
    )
    write_text(DATASET_DIR / "delta_force.yaml", yaml_content)

    manifest = [
        "Delta Force YOLOv8 dataset",
        f"images_total: {len(images)}",
        f"train_images: {len(splits['train'])}",
        f"val_images: {len(splits['val'])}",
        f"classes: {', '.join(CLASS_NAMES)}",
        f"source_frames: {FRAMES_DIR}",
        f"label_zip: {DEFAULT_LABEL_ZIP}",
        f"seed: {SEED}",
        "",
    ]
    write_text(DATASET_DIR / "dataset_manifest.txt", "\n".join(manifest))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate labels and build a YOLOv8 dataset.")
    parser.add_argument("--label-zip", type=Path, default=DEFAULT_LABEL_ZIP)
    parser.add_argument("--build", action="store_true", help="Create my_dataset after validation.")
    args = parser.parse_args()

    images = iter_images(FRAMES_DIR)
    labels = read_labels(args.label_zip)
    stats = validate(images, labels)

    print(f"images: {stats.image_count}")
    print(f"label txt files: {stats.label_count}")
    print(f"objects: {stats.object_count}")
    print(f"empty labels in zip: {stats.empty_label_count}")
    print(f"images without labels: {len(stats.missing_labels)}")
    print(f"labels without images: {len(stats.extra_labels)}")

    if stats.missing_labels:
        print("missing label images:")
        for stem in stats.missing_labels:
            print(f"  {stem}.jpg")
    if stats.extra_labels:
        print("extra label files:")
        for stem in stats.extra_labels:
            print(f"  {stem}.txt")

    if stats.extra_labels:
        raise SystemExit("Found labels without matching images; please fix before training.")

    if args.build:
        copy_dataset(images, labels)
        print(f"dataset written to: {DATASET_DIR}")
        print(f"yaml: {DATASET_DIR / 'delta_force.yaml'}")


if __name__ == "__main__":
    main()
