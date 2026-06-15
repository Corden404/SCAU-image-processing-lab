from __future__ import annotations

import re
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_MD = PROJECT_ROOT / "实验报告.md"
WORK_DIR = PROJECT_ROOT / "scau_work"
IMAGES_DIR = WORK_DIR / "images"

IMAGE_RE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")


def clean_image_name(index: int, source: Path) -> str:
    stem = re.sub(r"[^0-9A-Za-z_.-]+", "_", source.stem)
    suffix = source.suffix.lower() or ".png"
    return f"img_{index:03d}_{stem}{suffix}"


def copy_image(index: int, image_path: str) -> str:
    source = (PROJECT_ROOT / image_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Image referenced by report was not found: {image_path}")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    target_name = clean_image_name(index, source)
    target = IMAGES_DIR / target_name
    shutil.copy2(source, target)
    return f"images/{target_name}"


def convert_body(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    image_index = 1
    index = 0

    while index < len(lines):
        line = lines[index]

        if index == 0 and line.startswith("# 实验8"):
            index += 1
            continue
        if index == 1 and line.strip() == "---":
            index += 1
            continue

        match = IMAGE_RE.match(line)
        if match:
            image_path = match.group(2)
            copied_path = copy_image(image_index, image_path)
            image_index += 1

            output.append('::: {custom-style="SCAU_Image_Container"}')
            output.append(f"![]({copied_path})")
            output.append(":::")
            output.append("")

            next_index = index + 1
            while next_index < len(lines) and not lines[next_index].strip():
                next_index += 1

            if next_index < len(lines) and re.match(r"^图\s*\d+", lines[next_index].strip()):
                output.append('::: {custom-style="SCAU_Caption"}')
                output.append(lines[next_index].strip())
                output.append(":::")
                output.append("")
                index = next_index + 1
                continue

            index += 1
            continue

        output.append(line)
        index += 1

    return "\n".join(output).strip() + "\n"


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if IMAGES_DIR.exists():
        shutil.rmtree(IMAGES_DIR)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    source_text = SOURCE_MD.read_text(encoding="utf-8")
    body_text = convert_body(source_text)
    (WORK_DIR / "body.md").write_text(body_text, encoding="utf-8")

    (WORK_DIR / "abstract_cn.md").write_text(
        "\n".join(
            [
                '::: {custom-style="SCAU_Abstract_Title"}',
                "摘要",
                ":::",
                "",
                '::: {custom-style="SCAU_Abstract_Body"}',
                "本文档为实验报告预设生成的占位摘要模块，最终 DOCX 使用 report 预设，仅组装实验报告封面、目录和正文。",
                ":::",
                "",
                '::: {custom-style="SCAU_Keywords"}',
                '[关键词：]{custom-style="SCAU_Keyword_Label"} YOLOv8；目标检测；实验报告',
                ":::",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (WORK_DIR / "abstract_en.md").write_text(
        "\n".join(
            [
                '::: {custom-style="SCAU_English_Title"}',
                "Abstract",
                ":::",
                "",
                '::: {custom-style="SCAU_Abstract_En"}',
                "This placeholder module is generated only for module validation. The final document uses the report preset and does not include an English abstract.",
                ":::",
                "",
                '::: {custom-style="SCAU_Keywords"}',
                '[Key words:]{custom-style="SCAU_Keyword_Label"} YOLOv8; object detection; lab report',
                ":::",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote {WORK_DIR / 'body.md'}")
    print(f"Copied images to {IMAGES_DIR}")


if __name__ == "__main__":
    main()
