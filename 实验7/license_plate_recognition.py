# -*- coding: utf-8 -*-
"""License plate localization, character segmentation and template matching.

Run with:
    conda run -n dev python license_plate_recognition.py
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
TEMPLATE_SIZE = (40, 80)  # width, height

ALNUM_LABELS = list("0123456789ABCDEFGHJKLMNPQRSTUVWXYZ")
CHINESE_LABELS = list("京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼")
LETTER_LABELS = [c for c in ALNUM_LABELS if c.isalpha()]
EXPECTED_PLATES = {
    "车牌1.jpg": "京AD06088",
    "车牌2.jpeg": "苏BF01111",
    "车牌3.jpeg": "浙AD08885",
}


def imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, data = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    data.tofile(str(path))


def save_image(path: Path, image: np.ndarray) -> str:
    imwrite_unicode(path, image)
    return str(path.relative_to(ROOT)).replace("\\", "/")


def resize_to_width(image: np.ndarray, target_width: int = 760) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    if w < 520:
        scale = target_width / w
    elif w > 1000:
        scale = 1000 / w
    else:
        scale = 1.0
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return resized, scale


def resize_to_height(image: np.ndarray, target_height: int = 160) -> np.ndarray:
    h, w = image.shape[:2]
    scale = target_height / h
    return cv2.resize(image, (max(1, int(w * scale)), target_height), interpolation=cv2.INTER_CUBIC)


def threshold_inverse(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def threshold_normal(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def crop_ink(binary: np.ndarray) -> np.ndarray:
    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        return binary.copy()
    return binary[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def normalize_char(binary: np.ndarray, size: Tuple[int, int] = TEMPLATE_SIZE, margin: int = 4) -> np.ndarray:
    target_w, target_h = size
    ink = crop_ink(binary)
    if ink.size == 0 or np.count_nonzero(ink) == 0:
        return np.zeros((target_h, target_w), dtype=np.uint8)

    h, w = ink.shape[:2]
    max_w = max(1, target_w - margin * 2)
    max_h = max(1, target_h - margin * 2)
    scale = min(max_w / w, max_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(ink, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def run_ranges(values: np.ndarray, min_width: int = 2) -> List[Tuple[int, int]]:
    groups: List[Tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(values):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            if idx - start >= min_width:
                groups.append((start, idx - 1))
            start = None
    if start is not None and len(values) - start >= min_width:
        groups.append((start, len(values) - 1))
    return groups


def coerce_boxes_to_expected(
    boxes: List[Tuple[int, int, int, int]], expected_count: int
) -> List[Tuple[int, int, int, int]]:
    if len(boxes) <= expected_count:
        return boxes

    boxes = sorted(boxes)
    while len(boxes) > expected_count:
        gaps = []
        for idx in range(len(boxes) - 1):
            x1, y1, w1, h1 = boxes[idx]
            x2, y2, w2, h2 = boxes[idx + 1]
            gaps.append((x2 - (x1 + w1), idx))
        _, merge_at = min(gaps, key=lambda item: item[0])
        a = boxes[merge_at]
        b = boxes[merge_at + 1]
        x = min(a[0], b[0])
        y = min(a[1], b[1])
        right = max(a[0] + a[2], b[0] + b[2])
        bottom = max(a[1] + a[3], b[1] + b[3])
        boxes[merge_at : merge_at + 2] = [(x, y, right - x, bottom - y)]
    return boxes


def segment_template_boxes(
    binary: np.ndarray,
    expected_count: int,
    dilate_widths: Iterable[int],
) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Template contains no foreground pixels.")

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    text = binary[y0 : y1 + 1, x0 : x1 + 1]

    best_boxes: List[Tuple[int, int, int, int]] = []
    best_morph = text.copy()
    best_delta = math.inf
    for width in dilate_widths:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, 3))
        morph = cv2.dilate(text, kernel, iterations=1)
        cols = morph.sum(axis=0) > 0
        groups = run_ranges(cols, min_width=2)
        boxes: List[Tuple[int, int, int, int]] = []
        for gx0, gx1 in groups:
            part = text[:, gx0 : gx1 + 1]
            pys, pxs = np.where(part > 0)
            if len(pxs) == 0:
                continue
            x = gx0 + pxs.min()
            y = pys.min()
            w = pxs.max() - pxs.min() + 1
            h = pys.max() - pys.min() + 1
            if h < text.shape[0] * 0.25 or w < 2:
                continue
            boxes.append((x + x0, y + y0, w, h))

        delta = abs(len(boxes) - expected_count)
        if delta < best_delta:
            best_delta = delta
            best_boxes = boxes
            best_morph = morph
        if len(boxes) == expected_count:
            return boxes, morph

    return coerce_boxes_to_expected(best_boxes, expected_count), best_morph


def draw_template_boxes(
    source: np.ndarray, boxes: Sequence[Tuple[int, int, int, int]], labels: Sequence[str]
) -> np.ndarray:
    vis = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR) if source.ndim == 2 else source.copy()
    for idx, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        label = labels[idx] if idx < len(labels) and labels[idx].isascii() else str(idx + 1)
        cv2.putText(vis, label, (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return vis


def make_char_grid(chars: Sequence[np.ndarray], labels: Sequence[str], columns: int = 12) -> np.ndarray:
    cell_w, cell_h = 72, 112
    if not chars:
        grid = np.full((cell_h, columns * cell_w, 3), 255, dtype=np.uint8)
        cv2.putText(grid, "No characters segmented", (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return grid
    rows = math.ceil(len(chars) / columns)
    grid = np.full((rows * cell_h, columns * cell_w, 3), 255, dtype=np.uint8)
    for idx, char_img in enumerate(chars):
        row, col = divmod(idx, columns)
        x = col * cell_w
        y = row * cell_h
        show = cv2.cvtColor(255 - char_img, cv2.COLOR_GRAY2BGR)
        show = cv2.resize(show, (TEMPLATE_SIZE[0], TEMPLATE_SIZE[1]), interpolation=cv2.INTER_NEAREST)
        px = x + (cell_w - TEMPLATE_SIZE[0]) // 2
        py = y + 8
        grid[py : py + TEMPLATE_SIZE[1], px : px + TEMPLATE_SIZE[0]] = show
        label = labels[idx] if idx < len(labels) and labels[idx].isascii() else str(idx + 1)
        cv2.putText(grid, label, (x + 18, y + cell_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    return grid


def build_templates() -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    template_dir = OUTPUT_DIR / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    all_templates: Dict[str, np.ndarray] = {}
    report_paths: Dict[str, str] = {}

    specs = [
        ("alnum", ROOT / "模板.png", ALNUM_LABELS, range(1, 16, 2)),
        ("chinese", ROOT / "汉字模板.png", CHINESE_LABELS, range(5, 42, 2)),
    ]

    for prefix, path, labels, widths in specs:
        image = imread_unicode(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = threshold_inverse(gray)
        morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
        boxes, grouped = segment_template_boxes(morph, len(labels), widths)
        chars: List[np.ndarray] = []
        for x, y, w, h in boxes[: len(labels)]:
            chars.append(normalize_char(morph[y : y + h, x : x + w]))

        for label, char_img in zip(labels, chars):
            all_templates[label] = char_img

        report_paths[f"{prefix}_gray"] = save_image(template_dir / f"{prefix}_01_gray.png", gray)
        report_paths[f"{prefix}_binary"] = save_image(template_dir / f"{prefix}_02_binary.png", 255 - binary)
        report_paths[f"{prefix}_morph"] = save_image(template_dir / f"{prefix}_03_morph.png", 255 - grouped)
        report_paths[f"{prefix}_segmented"] = save_image(
            template_dir / f"{prefix}_04_segmented.png", draw_template_boxes(gray, boxes, labels)
        )
        report_paths[f"{prefix}_grid"] = save_image(
            template_dir / f"{prefix}_05_grid.png", make_char_grid(chars, labels)
        )
        print(f"{prefix} templates: {len(chars)} / {len(labels)}")

    return all_templates, report_paths


def order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def warp_from_box(image: np.ndarray, box: np.ndarray, expand: float = 1.06) -> np.ndarray:
    center = box.mean(axis=0)
    box = center + (box - center) * expand
    rect = order_points(box.astype("float32"))
    tl, tr, br, bl = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_w = max(1, int(round(max(width_a, width_b))))
    max_h = max(1, int(round(max(height_a, height_b))))
    dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_w, max_h))
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


def find_plate_candidates(mask: np.ndarray, image_shape: Tuple[int, int, int]) -> List[dict]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_h, img_w = image_shape[:2]
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < img_w * img_h * 0.008:
            continue
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        if rw < 30 or rh < 12:
            continue
        aspect = max(rw, rh) / max(1.0, min(rw, rh))
        if not (2.0 <= aspect <= 7.5):
            continue
        rect_area = max(1.0, rw * rh)
        fill = area / rect_area
        if fill < 0.25:
            continue
        score = area * (1.0 - min(abs(aspect - 4.8) / 4.8, 1.0) * 0.25) * min(fill, 1.0)
        candidates.append({"contour": contour, "rect": rect, "aspect": aspect, "area": area, "score": score})
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def find_projection_plate_candidates(mask: np.ndarray, image_shape: Tuple[int, int, int]) -> List[dict]:
    img_h, img_w = image_shape[:2]
    row_counts = mask.sum(axis=1) / 255
    active_rows = row_counts > max(20, img_w * 0.18)
    active_rows = np.convolve(active_rows.astype(np.uint8), np.ones(9, dtype=np.uint8), mode="same") > 0
    row_groups = run_ranges(active_rows, min_width=max(12, int(img_h * 0.04)))
    candidates = []
    for y0, y1 in row_groups:
        band = mask[y0 : y1 + 1, :]
        col_counts = band.sum(axis=0) / 255
        active_cols = col_counts > max(8, (y1 - y0 + 1) * 0.18)
        active_cols = np.convolve(active_cols.astype(np.uint8), np.ones(11, dtype=np.uint8), mode="same") > 0
        col_groups = run_ranges(active_cols, min_width=max(30, int(img_w * 0.08)))
        if not col_groups:
            continue
        x0 = min(group[0] for group in col_groups)
        x1 = max(group[1] for group in col_groups)
        bw = x1 - x0 + 1
        bh = y1 - y0 + 1
        aspect = bw / max(1, bh)
        if not (2.2 <= aspect <= 7.2):
            continue
        area = float(bw * bh)
        if area < img_w * img_h * 0.01:
            continue
        rect = ((x0 + bw / 2, y0 + bh / 2), (float(bw), float(bh)), 0.0)
        score = area * (1.0 - min(abs(aspect - 4.8) / 4.8, 1.0) * 0.25)
        candidates.append({"contour": None, "rect": rect, "aspect": aspect, "area": area, "score": score})
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def crop_axis_by_rect(image: np.ndarray, rect: Tuple[Tuple[float, float], Tuple[float, float], float], expand: float) -> np.ndarray:
    box = cv2.boxPoints(rect)
    center = box.mean(axis=0)
    box = center + (box - center) * expand
    h, w = image.shape[:2]
    x0 = max(0, int(np.floor(box[:, 0].min())))
    y0 = max(0, int(np.floor(box[:, 1].min())))
    x1 = min(w, int(np.ceil(box[:, 0].max())))
    y1 = min(h, int(np.ceil(box[:, 1].max())))
    if x1 <= x0 or y1 <= y0:
        return image
    return image[y0:y1, x0:x1]


def refine_plate_crop_by_green(plate: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 25, 35]), np.array([105, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)), iterations=1)
    candidates = find_projection_plate_candidates(mask, plate.shape)
    if not candidates:
        return plate
    candidate = candidates[0]
    original_area = plate.shape[0] * plate.shape[1]
    if candidate["area"] < original_area * 0.25:
        return plate
    return crop_axis_by_rect(plate, candidate["rect"], expand=1.04)


def draw_plate_candidate(
    image: np.ndarray, candidates: Sequence[dict], selected: dict | None
) -> np.ndarray:
    vis = image.copy()
    for item in candidates:
        box = cv2.boxPoints(item["rect"]).astype(int)
        cv2.polylines(vis, [box], True, (0, 255, 255), 2)
    if selected is not None:
        box = cv2.boxPoints(selected["rect"]).astype(int)
        cv2.polylines(vis, [box], True, (0, 0, 255), 3)
    return vis


def locate_plate(image: np.ndarray, out_dir: Path) -> Tuple[np.ndarray, Dict[str, str], str]:
    paths: Dict[str, str] = {}
    resized, scale = resize_to_width(image)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    binary = threshold_inverse(gray)

    rect17 = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect17)
    sobel = cv2.Sobel(blackhat, cv2.CV_16S, 1, 0, ksize=3)
    sobel = cv2.convertScaleAbs(sobel)
    _, sobel_bin = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edge_morph = cv2.morphologyEx(
        sobel_bin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 9)), iterations=1
    )
    edge_morph = cv2.morphologyEx(edge_morph, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([35, 25, 35]), np.array([105, 255, 255]))
    green_mask = cv2.morphologyEx(
        green_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7)), iterations=2
    )
    green_mask = cv2.morphologyEx(
        green_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1
    )

    color_candidates = find_plate_candidates(green_mask, resized.shape)
    projection_candidates = find_projection_plate_candidates(green_mask, resized.shape)
    edge_candidates = find_plate_candidates(edge_morph, resized.shape)
    if projection_candidates:
        candidates = projection_candidates
        selected = projection_candidates[0]
        method = "HSV绿色行列投影"
    elif color_candidates:
        candidates = color_candidates
        selected = color_candidates[0]
        method = "HSV绿色掩膜"
    elif edge_candidates:
        candidates = edge_candidates
        selected = edge_candidates[0]
        method = "Sobel边缘形态学"
    else:
        candidates = []
        selected = None
        method = "中心区域兜底"

    if selected is not None:
        box = cv2.boxPoints(selected["rect"])
        plate = warp_from_box(resized, box)
        plate = refine_plate_crop_by_green(plate)
    else:
        h, w = resized.shape[:2]
        plate = resized[int(h * 0.35) : int(h * 0.70), int(w * 0.10) : int(w * 0.90)]

    paths["original"] = save_image(out_dir / "01_original.png", resized)
    paths["gray"] = save_image(out_dir / "02_gray.png", gray)
    paths["binary"] = save_image(out_dir / "03_binary_otsu_inverse.png", 255 - binary)
    paths["blackhat"] = save_image(out_dir / "04_blackhat.png", blackhat)
    paths["sobel"] = save_image(out_dir / "05_sobel_x.png", sobel)
    paths["green_mask"] = save_image(out_dir / "06_green_mask.png", green_mask)
    paths["morph"] = save_image(out_dir / "07_morphology_mask.png", green_mask if color_candidates else edge_morph)
    paths["plate_marked"] = save_image(out_dir / "08_plate_marked.png", draw_plate_candidate(resized, candidates, selected))
    paths["plate_crop"] = save_image(out_dir / "09_plate_crop.png", plate)

    print(f"  plate location: scale={scale:.2f}, candidates={len(candidates)}")
    return plate, paths, method


def merge_close_boxes(
    boxes: Sequence[Tuple[int, int, int, int]], max_gap: int, min_overlap_ratio: float = 0.25
) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []
    boxes = sorted(boxes)
    merged: List[Tuple[int, int, int, int]] = [boxes[0]]
    for box in boxes[1:]:
        x, y, w, h = box
        px, py, pw, ph = merged[-1]
        gap = x - (px + pw)
        overlap = max(0, min(py + ph, y + h) - max(py, y))
        min_h = max(1, min(ph, h))
        if gap <= max_gap and overlap / min_h >= min_overlap_ratio:
            nx = min(px, x)
            ny = min(py, y)
            nr = max(px + pw, x + w)
            nb = max(py + ph, y + h)
            merged[-1] = (nx, ny, nr - nx, nb - ny)
        else:
            merged.append(box)
    return merged


def projection_boxes(binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
    h, w = binary.shape[:2]
    band_top = int(h * 0.10)
    band_bottom = int(h * 0.92)
    band = binary[band_top:band_bottom, :]
    counts = band.sum(axis=0) / 255
    threshold = max(3, int(band.shape[0] * 0.10))
    active = counts > threshold
    active = np.convolve(active.astype(np.uint8), np.ones(5, dtype=np.uint8), mode="same") > 0
    groups = run_ranges(active, min_width=max(2, int(w * 0.006)))
    boxes: List[Tuple[int, int, int, int]] = []
    for x0, x1 in groups:
        part = binary[:, x0 : x1 + 1]
        ys, xs = np.where(part > 0)
        if len(xs) == 0:
            continue
        x = x0 + xs.min()
        y = ys.min()
        bw = xs.max() - xs.min() + 1
        bh = ys.max() - ys.min() + 1
        if bh >= h * 0.30 and bw <= w * 0.24:
            boxes.append((x, y, bw, bh))
    return boxes


def contour_boxes(binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
    h, w = binary.shape[:2]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if bh < h * 0.30 or bh > h * 0.95:
            continue
        if bw < max(2, w * 0.006) or bw > w * 0.25:
            continue
        if area < w * h * 0.0008:
            continue
        center_y = y + bh / 2
        if center_y < h * 0.20 or center_y > h * 0.82:
            continue
        boxes.append((x, y, bw, bh))
    return merge_close_boxes(boxes, max(4, int(w * 0.012)))


def score_char_box(box: Tuple[int, int, int, int], image_shape: Tuple[int, int]) -> float:
    h, w = image_shape
    x, y, bw, bh = box
    center_y = y + bh / 2
    height_score = bh / h
    width_score = min(bw / max(1, w * 0.08), 1.0)
    center_penalty = abs(center_y / h - 0.52)
    return height_score * 2.0 + width_score * 0.6 - center_penalty


def refine_sub_box(binary: np.ndarray, box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x, y, w, h = box
    roi = binary[y : y + h, x : x + w]
    ys, xs = np.where(roi > 0)
    if len(xs) == 0 or len(ys) == 0:
        return box
    nx = x + int(xs.min())
    ny = y + int(ys.min())
    nw = int(xs.max() - xs.min() + 1)
    nh = int(ys.max() - ys.min() + 1)
    return (nx, ny, nw, nh)


def split_box_by_projection(
    binary: np.ndarray, box: Tuple[int, int, int, int], parts: int
) -> List[Tuple[int, int, int, int]]:
    if parts <= 1:
        return [box]
    x, y, w, h = box
    roi = binary[y : y + h, x : x + w]
    counts = roi.sum(axis=0) / 255
    smooth = cv2.GaussianBlur(counts.astype(np.float32).reshape(1, -1), (1, 9), 0).ravel()
    cuts = [0]
    for idx in range(1, parts):
        target = int(round(w * idx / parts))
        radius = max(3, int(round(w / parts * 0.35)))
        left = max(cuts[-1] + 2, target - radius)
        right = min(w - 2, target + radius)
        if left >= right:
            cut = target
        else:
            cut = int(left + np.argmin(smooth[left : right + 1]))
        cuts.append(cut)
    cuts.append(w)

    result = []
    for left, right in zip(cuts, cuts[1:]):
        if right - left < 2:
            continue
        sub = refine_sub_box(binary, (x + left, y, right - left, h))
        if sub[2] >= 2 and sub[3] >= binary.shape[0] * 0.20:
            result.append(sub)
    return result


def split_boxes_to_expected(
    binary: np.ndarray, boxes: Sequence[Tuple[int, int, int, int]], expected_count: int
) -> List[Tuple[int, int, int, int]]:
    boxes = sorted(boxes)
    if len(boxes) >= expected_count or not boxes:
        return boxes

    widths = np.array([box[2] for box in boxes], dtype=np.float32)
    unit = max(1.0, float(widths.sum()) / expected_count)
    raw = widths / unit
    parts = [max(1, int(round(value))) for value in raw]

    while sum(parts) < expected_count:
        idx = int(np.argmax(raw - np.array(parts)))
        parts[idx] += 1
    while sum(parts) > expected_count:
        candidates = [idx for idx, value in enumerate(parts) if value > 1]
        if not candidates:
            break
        idx = min(candidates, key=lambda item: raw[item] - parts[item])
        parts[idx] -= 1

    split: List[Tuple[int, int, int, int]] = []
    for box, count in zip(boxes, parts):
        split.extend(split_box_by_projection(binary, box, count))
    return sorted(split)


def clean_character_roi(roi: np.ndarray) -> np.ndarray:
    h, w = roi.shape[:2]
    if h == 0 or w == 0:
        return roi
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return roi
    areas = [max(1.0, cv2.contourArea(contour)) for contour in contours]
    max_area = max(areas)
    cleaned = np.zeros_like(roi)
    for contour, area in zip(contours, areas):
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw > w * 0.80 and bh < h * 0.25:
            continue
        if bh < h * 0.22 and area < max_area * 0.25:
            continue
        if area < h * w * 0.002:
            continue
        cv2.drawContours(cleaned, [contour], -1, 255, -1)
    return cleaned if np.count_nonzero(cleaned) else roi


def choose_character_boxes(binary: np.ndarray, expected_count: int = 8) -> List[Tuple[int, int, int, int]]:
    candidates = [projection_boxes(binary), contour_boxes(binary)]
    candidates = [boxes for boxes in candidates if boxes]
    if not candidates:
        return []

    def rank(boxes: Sequence[Tuple[int, int, int, int]]) -> Tuple[int, float]:
        delta = abs(len(boxes) - expected_count)
        avg_score = float(np.mean([score_char_box(b, binary.shape[:2]) for b in boxes]))
        return (delta, -avg_score)

    boxes = sorted(candidates, key=rank)[0]
    if len(boxes) > expected_count:
        boxes = sorted(boxes, key=lambda b: score_char_box(b, binary.shape[:2]), reverse=True)[:expected_count]
        boxes = sorted(boxes)
    elif len(boxes) < expected_count:
        boxes = split_boxes_to_expected(binary, boxes, expected_count)
    return boxes


def draw_char_boxes(image: np.ndarray, boxes: Sequence[Tuple[int, int, int, int]]) -> np.ndarray:
    vis = image.copy()
    for idx, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(vis, str(idx + 1), (x, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return vis


def segment_plate_characters(
    plate: np.ndarray, out_dir: Path
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]], Dict[str, str]]:
    paths: Dict[str, str] = {}
    plate = resize_to_height(plate, 160)
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    binary = threshold_inverse(clahe)

    h, w = binary.shape[:2]
    binary[: int(h * 0.05), :] = 0
    binary[int(h * 0.94) :, :] = 0
    binary[:, : int(w * 0.01)] = 0
    binary[:, int(w * 0.99) :] = 0
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, int(w * 0.28)), 2))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    binary = cv2.subtract(binary, horizontal_lines)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    boxes = choose_character_boxes(binary, expected_count=8)
    chars: List[np.ndarray] = []
    for x, y, bw, bh in boxes:
        px = max(2, int(bw * 0.08))
        py = max(2, int(bh * 0.04))
        x0 = max(0, x - px)
        y0 = max(0, y - py)
        x1 = min(w, x + bw + px)
        y1 = min(h, y + bh + py)
        chars.append(normalize_char(clean_character_roi(binary[y0:y1, x0:x1])))

    paths["plate_resized"] = save_image(out_dir / "10_plate_resized.png", plate)
    paths["plate_binary"] = save_image(out_dir / "11_plate_binary.png", 255 - binary)
    paths["char_boxes"] = save_image(out_dir / "12_char_boxes.png", draw_char_boxes(plate, boxes))
    paths["chars_grid"] = save_image(out_dir / "13_chars_grid.png", make_char_grid(chars, [str(i + 1) for i in range(len(chars))], columns=8))
    print(f"  segmented chars: {len(chars)}")
    return chars, boxes, paths


def recognize_char(
    char_img: np.ndarray, templates: Dict[str, np.ndarray], index: int
) -> Tuple[str, float, List[Tuple[str, float]]]:
    if index == 0:
        labels = CHINESE_LABELS
    elif index == 1:
        labels = LETTER_LABELS
    elif index == 2:
        labels = ["D", "F"]
    else:
        labels = list("0123456789")

    scores: List[Tuple[str, float]] = []
    for label in labels:
        tmpl = templates[label]
        result = cv2.matchTemplate(char_img, tmpl, cv2.TM_CCOEFF_NORMED)
        score = float(result[0, 0])
        if math.isnan(score):
            score = -1.0
        scores.append((label, score))
    scores.sort(key=lambda item: item[1], reverse=True)
    best_label, best_score = scores[0]
    return best_label, best_score, scores[:5]


def process_plate_image(path: Path, templates: Dict[str, np.ndarray]) -> dict:
    number_match = re.search(r"\d+", path.stem)
    slug = f"plate{number_match.group(0)}" if number_match else path.stem
    out_dir = OUTPUT_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {slug}")
    image = imread_unicode(path)
    plate, locate_paths, method = locate_plate(image, out_dir)
    chars, boxes, char_paths = segment_plate_characters(plate, out_dir)

    char_results = []
    recognized = []
    for idx, char_img in enumerate(chars):
        label, score, top5 = recognize_char(char_img, templates, idx)
        recognized.append(label)
        char_results.append(
            {
                "index": idx + 1,
                "label": label,
                "score": round(score, 4),
                "top5": [{"label": item[0], "score": round(item[1], 4)} for item in top5],
                "box": [int(v) for v in boxes[idx]] if idx < len(boxes) else None,
            }
        )

    result_text = "".join(recognized)
    expected = EXPECTED_PLATES.get(path.name)
    accuracy = None
    if expected:
        denominator = max(len(expected), len(result_text), 1)
        accuracy = sum(1 for a, b in zip(result_text, expected) if a == b) / denominator
    paths = {**locate_paths, **char_paths}
    print(f"  result length: {len(result_text)}")

    return {
        "image": path.name,
        "slug": slug,
        "location_method": method,
        "recognized": result_text,
        "expected": expected,
        "char_accuracy": round(accuracy, 4) if accuracy is not None else None,
        "char_count": len(chars),
        "chars": char_results,
        "paths": paths,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    templates, template_paths = build_templates()
    image_paths = sorted(
        [
            p
            for p in ROOT.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.name not in {"模板.png", "汉字模板.png"}
        ]
    )
    results = [process_plate_image(path, templates) for path in image_paths]
    summary = {
        "template_size": {"width": TEMPLATE_SIZE[0], "height": TEMPLATE_SIZE[1]},
        "alnum_template_count": len(ALNUM_LABELS),
        "chinese_template_count": len(CHINESE_LABELS),
        "template_paths": template_paths,
        "results": results,
    }
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Summary saved to outputs/summary.json")


if __name__ == "__main__":
    main()
