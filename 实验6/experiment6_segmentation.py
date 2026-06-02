from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


matplotlib.use("Agg")

IMAGE_PATH = Path("sample.png")
OUTPUT_DIR = Path("results")
MODEL_PATH = Path("yolov8n-seg.pt")
FIXED_THRESHOLD = 150
CANNY_THRESHOLDS = [(50, 100), (100, 200), (150, 300)]
YOLO_CONFIDENCE = 0.25


def save_gray(path: Path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), image)


def save_bgr(path: Path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), image)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def count_foreground(binary: np.ndarray) -> dict[str, float | int]:
    total = int(binary.size)
    foreground = int(np.count_nonzero(binary))
    return {
        "foreground_pixels": foreground,
        "total_pixels": total,
        "foreground_ratio": round(foreground / total, 4),
    }


def largest_contour_areas(contours: list[np.ndarray], top_n: int = 8) -> list[float]:
    areas = sorted((cv2.contourArea(c) for c in contours), reverse=True)
    return [round(float(area), 2) for area in areas[:top_n]]


def run_yolo_segmentation(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH}. Download it before running the deep learning task."
        )

    model = YOLO(str(MODEL_PATH))
    result = model.predict(
        source=str(IMAGE_PATH),
        conf=YOLO_CONFIDENCE,
        imgsz=640,
        device="cpu",
        verbose=False,
    )[0]

    overlay = result.plot()
    mask_canvas = img.copy()
    mask_layer = np.zeros_like(img)
    detections = []
    palette = [
        (0, 128, 255),
        (40, 200, 40),
        (255, 80, 80),
        (180, 80, 255),
        (255, 180, 30),
        (30, 200, 220),
        (210, 120, 20),
        (90, 180, 255),
    ]

    if result.masks is not None and result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for index, polygon in enumerate(result.masks.xy):
            if len(polygon) < 3:
                continue

            points = np.round(polygon).astype(np.int32).reshape((-1, 1, 2))
            color = palette[index % len(palette)]
            instance_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(instance_mask, [points], 255)
            cv2.fillPoly(mask_layer, [points], color)
            cv2.polylines(mask_canvas, [points], isClosed=True, color=color, thickness=3)

            class_id = int(classes[index])
            label = str(result.names[class_id])
            confidence = float(confidences[index])
            x1, y1, x2, y2 = boxes[index]
            cv2.rectangle(
                mask_canvas,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                2,
            )
            cv2.putText(
                mask_canvas,
                f"{label} {confidence:.2f}",
                (int(x1), max(24, int(y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
                cv2.LINE_AA,
            )

            detections.append(
                {
                    "class": label,
                    "confidence": round(confidence, 4),
                    "mask_pixels": int(np.count_nonzero(instance_mask)),
                    "bbox_xyxy": [
                        round(float(x1), 1),
                        round(float(y1), 1),
                        round(float(x2), 1),
                        round(float(y2), 1),
                    ],
                }
            )

    mask_canvas = cv2.addWeighted(mask_canvas, 0.65, mask_layer, 0.35, 0)
    class_counts: dict[str, int] = {}
    for item in detections:
        class_counts[item["class"]] = class_counts.get(item["class"], 0) + 1

    metrics = {
        "model": str(MODEL_PATH),
        "confidence_threshold": YOLO_CONFIDENCE,
        "detected_instances": len(detections),
        "class_counts": class_counts,
        "detections": detections,
    }
    return overlay, mask_canvas, metrics


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    img = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # The objects are darker than the bright desktop/background, so inverse
    # binary masks make the foreground easier to inspect.
    _, fixed_thresh = cv2.threshold(
        blur_gray, FIXED_THRESHOLD, 255, cv2.THRESH_BINARY_INV
    )
    adaptive_thresh = cv2.adaptiveThreshold(
        blur_gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5,
    )

    canny_results: dict[str, np.ndarray] = {}
    for low, high in CANNY_THRESHOLDS:
        canny_results[f"{low}_{high}"] = cv2.Canny(blur_gray, low, high)

    otsu_value, otsu_binary = cv2.threshold(
        blur_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel = np.ones((5, 5), np.uint8)
    region_mask = cv2.morphologyEx(otsu_binary, cv2.MORPH_OPEN, kernel, iterations=2)
    region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(
        region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    significant_contours = [
        contour for contour in contours if cv2.contourArea(contour) >= 1000
    ]
    contour_img = img.copy()
    cv2.drawContours(contour_img, significant_contours, -1, (0, 255, 0), 3)

    denoised = cv2.bilateralFilter(img, 9, 75, 75)
    denoised_gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    watershed_otsu, watershed_binary = cv2.threshold(
        denoised_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    opening = cv2.morphologyEx(watershed_binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(
        dist_transform, 0.45 * dist_transform.max(), 255, cv2.THRESH_BINARY
    )
    sure_fg_u8 = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg_u8)
    _, markers = cv2.connectedComponents(sure_fg_u8)
    markers = markers + 1
    markers[unknown == 255] = 0

    watershed_markers = cv2.watershed(denoised.copy(), markers.astype(np.int32))
    watershed_result = img.copy()
    watershed_result[watershed_markers == -1] = [0, 0, 255]
    yolo_overlay, yolo_masks, yolo_metrics = run_yolo_segmentation(img)

    save_bgr(OUTPUT_DIR / "01_original.png", img)
    save_gray(OUTPUT_DIR / "02_gray.png", gray)
    save_gray(OUTPUT_DIR / "03_fixed_threshold_inverse.png", fixed_thresh)
    save_gray(OUTPUT_DIR / "04_adaptive_threshold_inverse.png", adaptive_thresh)
    for key, value in canny_results.items():
        save_gray(OUTPUT_DIR / f"05_canny_{key}.png", value)
    save_gray(OUTPUT_DIR / "06_otsu_region_mask.png", region_mask)
    save_bgr(OUTPUT_DIR / "07_contours.png", contour_img)
    save_gray(OUTPUT_DIR / "08_watershed_binary.png", watershed_binary)
    save_gray(OUTPUT_DIR / "09_watershed_foreground.png", sure_fg_u8)
    save_gray(OUTPUT_DIR / "10_watershed_unknown.png", unknown)
    save_bgr(OUTPUT_DIR / "11_watershed_result.png", watershed_result)
    save_bgr(OUTPUT_DIR / "13_yolo_segmentation.png", yolo_overlay)
    save_bgr(OUTPUT_DIR / "14_yolo_masks.png", yolo_masks)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes[0, 0].imshow(bgr_to_rgb(img))
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(fixed_thresh, cmap="gray")
    axes[0, 1].set_title(f"Fixed threshold T={FIXED_THRESHOLD}")
    axes[0, 2].imshow(adaptive_thresh, cmap="gray")
    axes[0, 2].set_title("Adaptive threshold")
    axes[1, 0].imshow(canny_results["100_200"], cmap="gray")
    axes[1, 0].set_title("Canny 100/200")
    axes[1, 1].imshow(bgr_to_rgb(contour_img))
    axes[1, 1].set_title("Contours")
    axes[1, 2].imshow(bgr_to_rgb(watershed_result))
    axes[1, 2].set_title("Watershed")
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "12_summary.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(bgr_to_rgb(img))
    axes[0].set_title("Original")
    axes[1].imshow(bgr_to_rgb(watershed_result))
    axes[1].set_title("Traditional watershed")
    axes[2].imshow(bgr_to_rgb(yolo_masks))
    axes[2].set_title("YOLOv8 instance segmentation")
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "15_deep_learning_comparison.png", dpi=180)
    plt.close(fig)

    metrics = {
        "image": {
            "path": str(IMAGE_PATH),
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            "channels": int(img.shape[2]),
        },
        "fixed_threshold": {
            "threshold": FIXED_THRESHOLD,
            **count_foreground(fixed_thresh),
        },
        "adaptive_threshold": count_foreground(adaptive_thresh),
        "canny": {
            key: {
                "edge_pixels": int(np.count_nonzero(value)),
                "edge_ratio": round(float(np.count_nonzero(value) / value.size), 4),
            }
            for key, value in canny_results.items()
        },
        "contours": {
            "otsu_threshold": round(float(otsu_value), 2),
            "raw_count": int(len(contours)),
            "significant_count_area_ge_1000": int(len(significant_contours)),
            "largest_areas": largest_contour_areas(significant_contours),
        },
        "watershed": {
            "otsu_threshold": round(float(watershed_otsu), 2),
            "foreground_components": int(markers.max() - 1),
            "boundary_pixels": int(np.count_nonzero(watershed_markers == -1)),
        },
        "deep_learning_yolo": yolo_metrics,
        "outputs": sorted(str(path).replace("\\", "/") for path in OUTPUT_DIR.glob("*.png")),
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
