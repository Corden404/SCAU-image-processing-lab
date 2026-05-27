from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


IMAGE_PATH = Path("图片.jpg")
RESULT_DIR = Path("results")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return image


def save_image(name: str, image: np.ndarray) -> None:
    output_path = RESULT_DIR / name
    success, encoded = cv2.imencode(output_path.suffix, image)
    if not success:
        raise RuntimeError(f"Failed to encode image: {name}")
    encoded.tofile(str(output_path))


def plot_grid(items, columns: int, output_name: str, figsize=(12, 8)) -> None:
    rows = int(np.ceil(len(items) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for axis, item in zip(axes, items):
        title, image, is_color = item
        axis.set_title(title, fontsize=11)
        axis.axis("off")
        if is_color:
            axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            axis.imshow(image, cmap="gray", vmin=0, vmax=255)

    for axis in axes[len(items):]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(RESULT_DIR / output_name, dpi=200)
    plt.close(fig)


def foreground_pixels(image: np.ndarray) -> int:
    return int(np.count_nonzero(image))


def main() -> None:
    RESULT_DIR.mkdir(exist_ok=True)

    original = read_image(IMAGE_PATH)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    #笔画比背景更暗，使用反向 Otsu 阈值使区域成为白色前景。
    threshold_value, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    eroded = cv2.erode(binary, kernel_5, iterations=1)
    dilated = cv2.dilate(binary, kernel_5, iterations=1)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_5)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_5)

    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel_5)
    tophat = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel_5)
    # 黑帽用于突出暗孔洞和暗缝，二值图中的暗区域较细，使用较大核效果更明显。
    blackhat = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel_15)

    opening_3 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_3)
    opening_5 = opened
    opening_7 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_7)
    closing_3 = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_3)
    closing_5 = closed
    closing_7 = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_7)

    hitmiss_source = (binary // 255).astype(np.uint8)
    corner_kernels = [
        np.array([[1, 1, -1], [1, 1, -1], [-1, -1, -1]], dtype=np.int8),
        np.array([[-1, 1, 1], [-1, 1, 1], [-1, -1, -1]], dtype=np.int8),
        np.array([[-1, -1, -1], [1, 1, -1], [1, 1, -1]], dtype=np.int8),
        np.array([[-1, -1, -1], [-1, 1, 1], [-1, 1, 1]], dtype=np.int8),
    ]
    hitmiss_raw = np.zeros_like(hitmiss_source)
    for hitmiss_kernel in corner_kernels:
        detected = cv2.morphologyEx(
            hitmiss_source, cv2.MORPH_HITMISS, hitmiss_kernel
        )
        hitmiss_raw = cv2.bitwise_or(hitmiss_raw, detected)

    hitmiss = hitmiss_raw * 255
    hitmiss_display = cv2.dilate(hitmiss, kernel_5, iterations=1)
    hitmiss_overlay = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    hitmiss_overlay[hitmiss_display > 0] = (0, 0, 255)

    images = {
        "01_original.png": original,
        "02_gray.png": gray,
        "03_binary.png": binary,
        "04_eroded.png": eroded,
        "05_dilated.png": dilated,
        "06_opened.png": opened,
        "07_closed.png": closed,
        "08_gradient.png": gradient,
        "09_tophat.png": tophat,
        "10_blackhat.png": blackhat,
        "11_hitmiss.png": hitmiss,
        "12_hitmiss_overlay.png": hitmiss_overlay,
    }
    for filename, image in images.items():
        save_image(filename, image)

    plot_grid(
        [
            ("原图", original, True),
            ("灰度图", gray, False),
            ("二值前景图", binary, False),
            ("腐蚀", eroded, False),
            ("膨胀", dilated, False),
            ("开运算", opened, False),
            ("闭运算", closed, False),
        ],
        columns=4,
        output_name="basic_morphology_comparison.png",
        figsize=(13, 7),
    )

    plot_grid(
        [
            ("二值前景图", binary, False),
            ("形态学梯度", gradient, False),
            ("顶帽运算", tophat, False),
            ("黑帽运算", blackhat, False),
        ],
        columns=4,
        output_name="advanced_morphology_comparison.png",
        figsize=(13, 4),
    )

    plot_grid(
        [
            ("开运算 3x3", opening_3, False),
            ("开运算 5x5", opening_5, False),
            ("开运算 7x7", opening_7, False),
            ("闭运算 3x3", closing_3, False),
            ("闭运算 5x5", closing_5, False),
            ("闭运算 7x7", closing_7, False),
        ],
        columns=3,
        output_name="kernel_size_effect.png",
        figsize=(10, 7),
    )

    plot_grid(
        [
            ("二值前景图", binary, False),
            ("击中击不中结果", hitmiss_display, False),
            ("检测位置叠加图", hitmiss_overlay, True),
        ],
        columns=3,
        output_name="hitmiss_comparison.png",
        figsize=(11, 4),
    )

    stats = {
        "Otsu threshold": float(threshold_value),
        "Binary": foreground_pixels(binary),
        "Erosion": foreground_pixels(eroded),
        "Dilation": foreground_pixels(dilated),
        "Opening": foreground_pixels(opened),
        "Closing": foreground_pixels(closed),
        "Gradient": foreground_pixels(gradient),
        "Top-hat": foreground_pixels(tophat),
        "Black-hat": foreground_pixels(blackhat),
        "Hit-or-miss": foreground_pixels(hitmiss),
    }

    kernel_stats = {
        "Opening 3x3": foreground_pixels(opening_3),
        "Opening 5x5": foreground_pixels(opening_5),
        "Opening 7x7": foreground_pixels(opening_7),
        "Closing 3x3": foreground_pixels(closing_3),
        "Closing 5x5": foreground_pixels(closing_5),
        "Closing 7x7": foreground_pixels(closing_7),
    }

    summary_lines = ["Experiment 4 morphology processing finished"]
    for key, value in stats.items():
        summary_lines.append(f"{key}: {value}")
    summary_lines.append("")
    summary_lines.append("Kernel size effect")
    for key, value in kernel_stats.items():
        summary_lines.append(f"{key}: {value}")
    summary_lines.append(f"Result directory: {RESULT_DIR.name}")

    (RESULT_DIR / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
