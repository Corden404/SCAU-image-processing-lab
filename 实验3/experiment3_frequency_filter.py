from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter


IMAGE_PATH = Path("学校.jpg")
OUT_DIR = Path("results")
CUTOFF = 70
HIGH_BOOST_AMOUNT = 1.4
SPATIAL_BLUR_RADIUS = 3.0


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    min_value = arr.min()
    max_value = arr.max()
    if max_value == min_value:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - min_value) / (max_value - min_value) * 255).astype(np.uint8)


def save_gray(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(normalize_to_uint8(arr)).save(path)


def circular_mask(shape: tuple[int, int], cutoff: int, pass_type: str) -> np.ndarray:
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((y - crow) ** 2 + (x - ccol) ** 2)
    low_pass = (distance <= cutoff).astype(np.float64)
    if pass_type == "low":
        return low_pass
    if pass_type == "high":
        return 1.0 - low_pass
    raise ValueError("pass_type must be 'low' or 'high'")


def gaussian_frequency_mask(shape: tuple[int, int], cutoff: int, pass_type: str) -> np.ndarray:
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    distance2 = (y - crow) ** 2 + (x - ccol) ** 2
    low_pass = np.exp(-distance2 / (2 * cutoff**2))
    if pass_type == "low":
        return low_pass
    if pass_type == "high":
        return 1.0 - low_pass
    raise ValueError("pass_type must be 'low' or 'high'")


def apply_frequency_filter(f_shift: np.ndarray, mask: np.ndarray) -> np.ndarray:
    filtered_shift = f_shift * mask
    filtered = np.fft.ifft2(np.fft.ifftshift(filtered_shift))
    return np.real(filtered)


def save_figure(path: Path, images: list[tuple[str, np.ndarray]], cols: int = 3) -> None:
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 4.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, image) in zip(axes, images):
        ax.imshow(image, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    for ax in axes[len(images) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    image = Image.open(IMAGE_PATH).convert("L")
    gray = np.asarray(image, dtype=np.float64)

    f = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f)
    magnitude_spectrum = np.log1p(np.abs(f_shift))
    phase_spectrum = np.angle(f_shift)

    low_mask = circular_mask(gray.shape, CUTOFF, "low")
    high_mask = circular_mask(gray.shape, CUTOFF, "high")
    gaussian_low_mask = gaussian_frequency_mask(gray.shape, CUTOFF, "low")
    gaussian_high_mask = gaussian_frequency_mask(gray.shape, CUTOFF, "high")

    freq_low = apply_frequency_filter(f_shift, low_mask)
    freq_high_component = apply_frequency_filter(f_shift, high_mask)
    freq_high_boost = np.clip(gray + HIGH_BOOST_AMOUNT * freq_high_component, 0, 255)

    gaussian_freq_low = apply_frequency_filter(f_shift, gaussian_low_mask)
    gaussian_freq_high_component = apply_frequency_filter(f_shift, gaussian_high_mask)
    gaussian_freq_high_boost = np.clip(
        gray + HIGH_BOOST_AMOUNT * gaussian_freq_high_component, 0, 255
    )

    spatial_low = np.asarray(
        image.filter(ImageFilter.GaussianBlur(radius=SPATIAL_BLUR_RADIUS)),
        dtype=np.float64,
    )
    spatial_high_component = gray - spatial_low
    spatial_high_boost = np.clip(gray + HIGH_BOOST_AMOUNT * spatial_high_component, 0, 255)

    save_gray(OUT_DIR / "01_original_gray.png", gray)
    save_gray(OUT_DIR / "02_magnitude_spectrum.png", magnitude_spectrum)
    save_gray(OUT_DIR / "03_phase_spectrum.png", phase_spectrum)
    save_gray(OUT_DIR / "04_low_pass_mask.png", low_mask)
    save_gray(OUT_DIR / "05_high_pass_mask.png", high_mask)
    save_gray(OUT_DIR / "06_frequency_low_pass.png", freq_low)
    save_gray(OUT_DIR / "07_frequency_high_pass_detail.png", np.abs(freq_high_component))
    save_gray(OUT_DIR / "08_frequency_high_boost.png", freq_high_boost)
    save_gray(OUT_DIR / "09_spatial_low_pass.png", spatial_low)
    save_gray(OUT_DIR / "10_spatial_high_pass_detail.png", np.abs(spatial_high_component))
    save_gray(OUT_DIR / "11_spatial_high_boost.png", spatial_high_boost)
    save_gray(OUT_DIR / "12_gaussian_frequency_low_pass.png", gaussian_freq_low)
    save_gray(
        OUT_DIR / "13_gaussian_frequency_high_boost.png",
        gaussian_freq_high_boost,
    )

    save_figure(
        OUT_DIR / "A_fft_spectrum.png",
        [
            ("Original grayscale", gray),
            ("Magnitude spectrum log(1+|F|)", magnitude_spectrum),
            ("Phase spectrum", phase_spectrum),
        ],
        cols=3,
    )
    save_figure(
        OUT_DIR / "B_frequency_masks_and_results.png",
        [
            ("Original grayscale", gray),
            (f"Ideal low-pass mask D0={CUTOFF}", low_mask),
            (f"Ideal high-pass mask D0={CUTOFF}", high_mask),
            ("Frequency low-pass result", freq_low),
            ("Frequency high-pass detail", np.abs(freq_high_component)),
            ("Frequency high-boost result", freq_high_boost),
        ],
        cols=3,
    )
    save_figure(
        OUT_DIR / "C_frequency_vs_spatial.png",
        [
            ("Original grayscale", gray),
            ("Frequency low-pass", freq_low),
            ("Spatial Gaussian blur", spatial_low),
            ("Frequency high-boost", freq_high_boost),
            ("Spatial unsharp mask", spatial_high_boost),
            ("Gaussian frequency high-boost", gaussian_freq_high_boost),
        ],
        cols=3,
    )
    save_figure(
        OUT_DIR / "D_optional_gaussian_frequency_filter.png",
        [
            ("Original grayscale", gray),
            ("Gaussian low-pass mask", gaussian_low_mask),
            ("Gaussian high-pass mask", gaussian_high_mask),
            ("Gaussian frequency low-pass", gaussian_freq_low),
            ("Gaussian frequency high-boost", gaussian_freq_high_boost),
        ],
        cols=3,
    )

    print(f"Input image size: {image.width}x{image.height}")
    print(f"Output directory: {OUT_DIR}")
    print(f"Cutoff radius D0: {CUTOFF}")


if __name__ == "__main__":
    main()
