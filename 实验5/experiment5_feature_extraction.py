from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


IMAGE_PATH = Path("cat.png")
OUTPUT_DIR = Path("outputs")


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def imread_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    imwrite(path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def save_gray(path: Path, image_gray: np.ndarray) -> None:
    imwrite(path, image_gray)


def save_channel_grid(
    path: Path,
    rows: list[tuple[str, list[tuple[str, np.ndarray]]]],
) -> None:
    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 9), constrained_layout=True)
    for row_idx, (space_name, channels) in enumerate(rows):
        for col_idx, (channel_name, channel) in enumerate(channels):
            ax = axes[row_idx][col_idx]
            ax.imshow(channel, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{space_name} - {channel_name}")
            ax.axis("off")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_four_panel(path: Path, panels: list[tuple[str, np.ndarray, str | None]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 11), constrained_layout=True)
    for ax, (title, image, cmap) in zip(axes.flat, panels):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def normalize_abs_to_uint8(values: np.ndarray) -> np.ndarray:
    abs_values = np.abs(values)
    return cv2.normalize(abs_values, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def compute_lbp(gray: np.ndarray) -> np.ndarray:
    padded = np.pad(gray, pad_width=1, mode="edge")
    center = padded[1:-1, 1:-1]
    offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
    ]
    lbp = np.zeros_like(gray, dtype=np.uint8)
    for bit, (dy, dx) in enumerate(offsets):
        neighbor = padded[1 + dy : 1 + dy + gray.shape[0], 1 + dx : 1 + dx + gray.shape[1]]
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)
    return lbp


def save_lbp_results(
    gray: np.ndarray,
    lbp: np.ndarray,
    hist: np.ndarray,
    output_dir: Path,
) -> None:
    save_gray(output_dir / "lbp_gray.png", gray)
    save_gray(output_dir / "lbp_feature.png", lbp)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("灰度图")
    axes[0].axis("off")
    axes[1].imshow(lbp, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("LBP 特征图")
    axes[1].axis("off")
    axes[2].bar(np.arange(256), hist, width=1.0, color="#3b6ea8")
    axes[2].set_title("归一化 LBP 直方图")
    axes[2].set_xlabel("LBP 编码")
    axes[2].set_ylabel("频率")
    axes[2].set_xlim(0, 255)
    fig.savefig(output_dir / "lbp_summary.png", dpi=180)
    plt.close(fig)


def compute_glcm(
    gray: np.ndarray,
    levels: int = 16,
    distance: int = 1,
) -> tuple[np.ndarray, dict[str, dict[str, float]], np.ndarray]:
    quantized = (gray.astype(np.uint16) * levels // 256).astype(np.uint8)
    angles = {
        "0°": (0, distance),
        "45°": (-distance, distance),
        "90°": (-distance, 0),
        "135°": (-distance, -distance),
    }
    glcms = np.zeros((len(angles), levels, levels), dtype=np.float64)
    features: dict[str, dict[str, float]] = {}
    row_idx, col_idx = np.indices((levels, levels))

    for idx, (angle_name, (dy, dx)) in enumerate(angles.items()):
        y_src_start = max(0, -dy)
        y_src_end = gray.shape[0] - max(0, dy)
        x_src_start = max(0, -dx)
        x_src_end = gray.shape[1] - max(0, dx)
        source = quantized[y_src_start:y_src_end, x_src_start:x_src_end]
        neighbor = quantized[
            y_src_start + dy : y_src_end + dy,
            x_src_start + dx : x_src_end + dx,
        ]

        matrix = np.zeros((levels, levels), dtype=np.float64)
        np.add.at(matrix, (source.ravel(), neighbor.ravel()), 1)
        matrix += matrix.T
        total = matrix.sum()
        if total > 0:
            matrix /= total
        glcms[idx] = matrix

        diff = row_idx - col_idx
        contrast = float(np.sum(matrix * diff**2))
        dissimilarity = float(np.sum(matrix * np.abs(diff)))
        homogeneity = float(np.sum(matrix / (1.0 + diff**2)))
        asm = float(np.sum(matrix**2))
        energy = float(np.sqrt(asm))
        row_mean = float(np.sum(row_idx * matrix))
        col_mean = float(np.sum(col_idx * matrix))
        row_std = float(np.sqrt(np.sum(((row_idx - row_mean) ** 2) * matrix)))
        col_std = float(np.sqrt(np.sum(((col_idx - col_mean) ** 2) * matrix)))
        if row_std > 1e-12 and col_std > 1e-12:
            correlation = float(
                np.sum((row_idx - row_mean) * (col_idx - col_mean) * matrix) / (row_std * col_std)
            )
        else:
            correlation = 0.0
        nonzero = matrix[matrix > 0]
        entropy = float(-np.sum(nonzero * np.log2(nonzero)))
        features[angle_name] = {
            "contrast": contrast,
            "dissimilarity": dissimilarity,
            "homogeneity": homogeneity,
            "energy": energy,
            "correlation": correlation,
            "entropy": entropy,
        }

    return glcms, features, quantized


def save_glcm_results(
    gray: np.ndarray,
    glcms: np.ndarray,
    quantized: np.ndarray,
    output_dir: Path,
) -> None:
    save_gray(output_dir / "glcm_quantized.png", (quantized * (255 // 15)).astype(np.uint8))

    angle_names = ["0°", "45°", "90°", "135°"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes[0][0].imshow(gray, cmap="gray")
    axes[0][0].set_title("灰度图")
    axes[0][0].axis("off")
    axes[0][1].imshow(quantized, cmap="gray", vmin=0, vmax=15)
    axes[0][1].set_title("16 级量化灰度图")
    axes[0][1].axis("off")

    heatmap_axes = [axes[0][2], axes[1][0], axes[1][1], axes[1][2]]
    for ax, angle_name, matrix in zip(heatmap_axes, angle_names, glcms):
        image = ax.imshow(matrix, cmap="magma")
        ax.set_title(f"GLCM {angle_name}")
        ax.set_xlabel("邻域灰度级")
        ax.set_ylabel("中心灰度级")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(output_dir / "glcm_summary.png", dpi=180)
    plt.close(fig)


def summarize_glcm_features(features: dict[str, dict[str, float]]) -> dict[str, float]:
    metric_names = next(iter(features.values())).keys()
    return {
        metric_name: round(
            float(np.mean([angle_features[metric_name] for angle_features in features.values()])),
            6,
        )
        for metric_name in metric_names
    }


def make_haar_outputs(gray: np.ndarray, output_dir: Path) -> dict[str, float | int]:
    gray_float = gray.astype(np.float32) / 255.0
    window_size = 48
    half = window_size // 2
    half_columns = np.ones((window_size, half), dtype=np.float32)
    half_rows = np.ones((half, window_size), dtype=np.float32)
    quadrant = np.ones((half, half), dtype=np.float32)

    horizontal_kernel = np.hstack((-half_columns, half_columns)) / (window_size * window_size)
    vertical_kernel = np.vstack((-half_rows, half_rows)) / (window_size * window_size)
    checker_kernel = np.block([[quadrant, -quadrant], [-quadrant, quadrant]]) / (window_size * window_size)
    kernels = {
        "horizontal": horizontal_kernel,
        "vertical": vertical_kernel,
        "checker": checker_kernel,
    }

    responses = {
        name: cv2.filter2D(gray_float, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
        for name, kernel in kernels.items()
    }
    response_images = {
        name: cv2.applyColorMap(normalize_abs_to_uint8(response), cv2.COLORMAP_TURBO)
        for name, response in responses.items()
    }
    for name, response_image in response_images.items():
        imwrite(output_dir / f"haar_{name}_response.png", response_image)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes[0][0].imshow(gray, cmap="gray")
    axes[0][0].set_title("灰度图")
    axes[0][0].axis("off")
    for ax, title, name in [
        (axes[0][1], "Haar 左右双矩形响应", "horizontal"),
        (axes[1][0], "Haar 上下双矩形响应", "vertical"),
        (axes[1][1], "Haar 四矩形纹理响应", "checker"),
    ]:
        ax.imshow(cv2.cvtColor(response_images[name], cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(output_dir / "haar_summary.png", dpi=180)
    plt.close(fig)

    metrics: dict[str, float | int] = {"haar_window_size": window_size}
    for name, response in responses.items():
        abs_response = np.abs(response)
        metrics[f"haar_{name}_mean_abs"] = round(float(np.mean(abs_response)), 6)
        metrics[f"haar_{name}_p95_abs"] = round(float(np.percentile(abs_response, 95)), 6)
        metrics[f"haar_{name}_max_abs"] = round(float(np.max(abs_response)), 6)
    return metrics


def make_sift_outputs(bgr: np.ndarray, gray: np.ndarray, output_dir: Path) -> dict[str, int | float]:
    height, width = gray.shape
    center = (width / 2.0, height / 2.0)
    affine = cv2.getRotationMatrix2D(center, angle=18, scale=0.92)
    affine[0, 2] += 70
    affine[1, 2] -= 45
    transformed = cv2.warpAffine(
        bgr,
        affine,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    transformed_gray = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=1600)
    keypoints_1, descriptors_1 = sift.detectAndCompute(gray, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(transformed_gray, None)
    if descriptors_1 is None or descriptors_2 is None:
        raise RuntimeError("SIFT did not find enough descriptors.")

    original_keypoints = cv2.drawKeypoints(
        bgr,
        keypoints_1,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    transformed_keypoints = cv2.drawKeypoints(
        transformed,
        keypoints_2,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    imwrite(output_dir / "sift_keypoints_original.png", original_keypoints)
    imwrite(output_dir / "sift_keypoints_transformed.png", transformed_keypoints)
    imwrite(output_dir / "sift_transformed.png", transformed)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = matcher.knnMatch(descriptors_1, descriptors_2, k=2)
    good_matches = []
    for match_pair in raw_matches:
        if len(match_pair) < 2:
            continue
        m, n = match_pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    good_matches = sorted(good_matches, key=lambda m: m.distance)

    match_view = cv2.drawMatches(
        bgr,
        keypoints_1,
        transformed,
        keypoints_2,
        good_matches[:80],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    imwrite(output_dir / "sift_matches.png", match_view)

    inlier_count = 0
    aligned = np.zeros_like(bgr)
    if len(good_matches) >= 4:
        src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        homography, inlier_mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if homography is not None:
            aligned = cv2.warpPerspective(transformed, homography, (width, height))
            inlier_count = int(inlier_mask.sum()) if inlier_mask is not None else 0
    imwrite(output_dir / "sift_aligned.png", aligned)

    diff = cv2.absdiff(bgr, aligned)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_vis = cv2.applyColorMap(cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_TURBO)
    save_four_panel(
        output_dir / "sift_alignment_summary.png",
        [
            ("原始图像", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), None),
            ("旋转 + 缩放 + 平移后图像", cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB), None),
            ("SIFT 匹配对齐结果", cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB), None),
            ("对齐差异热力图", cv2.cvtColor(diff_vis, cv2.COLOR_BGR2RGB), None),
        ],
    )

    return {
        "sift_keypoints_original": len(keypoints_1),
        "sift_keypoints_transformed": len(keypoints_2),
        "sift_raw_matches": len(raw_matches),
        "sift_good_matches": len(good_matches),
        "sift_inliers": inlier_count,
        "sift_transform_angle_degree": 18,
        "sift_transform_scale": 0.92,
        "sift_translate_x": 70,
        "sift_translate_y": -45,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bgr = imread_bgr(IMAGE_PATH)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = bgr.shape[:2]
    save_rgb(OUTPUT_DIR / "original.png", rgb)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)

    r, g, b = cv2.split(rgb)
    h, s, v = cv2.split(hsv)
    lab_l, lab_a, lab_b = cv2.split(lab)
    lab_a_display = cv2.normalize(lab_a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lab_b_display = cv2.normalize(lab_b, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    save_channel_grid(
        OUTPUT_DIR / "color_space_channels.png",
        [
            ("RGB", [("R", r), ("G", g), ("B", b)]),
            ("HSV", [("H", h), ("S", s), ("V", v)]),
            ("Lab", [("L", lab_l), ("a 归一化", lab_a_display), ("b 归一化", lab_b_display)]),
        ],
    )

    # Hue selects the yellow-orange fur, while S and V suppress gray road and dark brick areas.
    lower_yellow = np.array([12, 55, 120], dtype=np.uint8)
    upper_yellow = np.array([38, 255, 255], dtype=np.uint8)
    raw_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    opened_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    cleaned_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    target_mask = np.zeros_like(cleaned_mask)
    contour_area = 0.0
    centroid = [0, 0]
    bounding_box = [0, 0, 0, 0]
    if contours:
        largest = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(largest))
        cv2.drawContours(target_mask, [largest], -1, 255, thickness=cv2.FILLED)
        moments = cv2.moments(largest)
        if moments["m00"] != 0:
            centroid = [int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])]
        bounding_box = [int(value) for value in cv2.boundingRect(largest)]

    yellow_target_bgr = cv2.bitwise_and(bgr, bgr, mask=target_mask)
    contour_overlay = rgb.copy()
    if contours:
        cv2.drawContours(contour_overlay, [largest], -1, (255, 0, 0), 5)
        cv2.circle(contour_overlay, tuple(centroid), 12, (0, 0, 255), -1)
        x, y, w_box, h_box = bounding_box
        cv2.rectangle(contour_overlay, (x, y), (x + w_box, y + h_box), (0, 255, 0), 4)

    save_gray(OUTPUT_DIR / "yellow_mask_raw.png", raw_mask)
    save_gray(OUTPUT_DIR / "yellow_mask_cleaned.png", cleaned_mask)
    imwrite(OUTPUT_DIR / "yellow_target.png", yellow_target_bgr)
    save_rgb(OUTPUT_DIR / "yellow_contour_overlay.png", contour_overlay)
    save_four_panel(
        OUTPUT_DIR / "yellow_extraction_summary.png",
        [
            ("原始 HSV 阈值掩膜", raw_mask, "gray"),
            ("开闭运算后掩膜", cleaned_mask, "gray"),
            ("黄色目标提取", cv2.cvtColor(yellow_target_bgr, cv2.COLOR_BGR2RGB), None),
            ("轮廓、外接框与重心", contour_overlay, None),
        ],
    )

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lbp = compute_lbp(gray)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    lbp_hist = lbp_hist.astype(np.float64)
    lbp_hist /= lbp_hist.sum()
    save_lbp_results(gray, lbp, lbp_hist, OUTPUT_DIR)

    glcms, glcm_features, glcm_quantized = compute_glcm(gray)
    save_glcm_results(gray, glcms, glcm_quantized, OUTPUT_DIR)
    glcm_mean_features = summarize_glcm_features(glcm_features)
    haar_metrics = make_haar_outputs(gray, OUTPUT_DIR)

    sift_metrics = make_sift_outputs(bgr, gray, OUTPUT_DIR)
    top_lbp_bins = np.argsort(lbp_hist)[-5:][::-1]

    metrics = {
        "image": IMAGE_PATH.name,
        "image_width": width,
        "image_height": height,
        "yellow_hsv_lower": lower_yellow.tolist(),
        "yellow_hsv_upper": upper_yellow.tolist(),
        "yellow_raw_mask_pixels": int(np.count_nonzero(raw_mask)),
        "yellow_cleaned_mask_pixels": int(np.count_nonzero(cleaned_mask)),
        "yellow_target_pixels": int(np.count_nonzero(target_mask)),
        "yellow_contour_area": round(contour_area, 2),
        "yellow_centroid": centroid,
        "yellow_bounding_box_xywh": bounding_box,
        "lbp_histogram_sum": float(lbp_hist.sum()),
        "lbp_top_bins": [
            {"bin": int(idx), "frequency": round(float(lbp_hist[idx]), 6)} for idx in top_lbp_bins
        ],
        "glcm_levels": 16,
        "glcm_distance": 1,
        "glcm_mean_features": glcm_mean_features,
        "glcm_direction_features": {
            angle_name: {
                metric_name: round(float(value), 6)
                for metric_name, value in angle_features.items()
            }
            for angle_name, angle_features in glcm_features.items()
        },
        **haar_metrics,
        **sift_metrics,
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
