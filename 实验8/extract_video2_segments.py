from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent
VIDEO_PATH = PROJECT_ROOT / "videos" / "2.mp4"
OUTPUT_DIR = PROJECT_ROOT / "extracted_frames" / "video2_segments_fps2"
TARGET_FPS = 2

SEGMENTS = [
    ("00:05", "00:20"),
]


def write_jpg(path: Path, frame) -> None:
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError(f"Cannot encode frame for {path}")
    encoded.tofile(str(path))


def parse_time(value: str) -> float:
    parts = [float(part) for part in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Unsupported time format: {value}")


def main() -> None:
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(VIDEO_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        raise RuntimeError("Cannot read video FPS.")

    step = max(1, round(source_fps / TARGET_FPS))
    saved = 0

    for segment_id, (start_text, end_text) in enumerate(SEGMENTS, start=1):
        start_sec = parse_time(start_text)
        end_sec = parse_time(end_text)
        start_frame = round(start_sec * source_fps)
        end_frame = round(end_sec * source_fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        segment_saved = 0

        while current_frame <= end_frame:
            ok, frame = cap.read()
            if not ok:
                break

            if (current_frame - start_frame) % step == 0:
                frame_sec = current_frame / source_fps
                output_path = OUTPUT_DIR / (
                    f"video2_s{segment_id:02d}_t{frame_sec:08.3f}_f{current_frame:06d}.jpg"
                )
                if not output_path.exists():
                    write_jpg(output_path, frame)
                    saved += 1
                    segment_saved += 1

            current_frame += 1

        print(
            f"segment {segment_id}: {start_text}-{end_text}, "
            f"frames {start_frame}-{end_frame}, saved {segment_saved}"
        )

    cap.release()
    print(f"source_fps={source_fps:.3f}, target_fps={TARGET_FPS}, total_saved={saved}")
    print(f"output_dir={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
