import argparse
import csv
import json
from collections import Counter
from pathlib import Path


REQUIRED_COLUMNS = [
    "split",
    "image_id",
    "image_path",
    "label",
    "sex",
    "age_group",
    "anatom_site_general",
    "lesion_id",
]


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []


def build_image_index(image_root):
    if not image_root.exists():
        return {}

    image_index = {}
    for path in image_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = path.stem.lower()
        image_index.setdefault(stem, path)
        if stem.endswith("_downsampled"):
            image_index.setdefault(stem.removesuffix("_downsampled"), path)
    return image_index


def normalize_image_id(value):
    image_id = str(value or "").strip()
    if not image_id:
        return ""
    return Path(image_id).stem.lower()


def resolve_image_path(row, project_root, image_dir, image_ext, image_index):
    csv_path = row.get("image_path", "").strip()
    if csv_path:
        path = Path(csv_path)
        if not path.is_absolute():
            path = project_root / path
        if path.exists():
            return path

    image_id = row.get("image_id", "").strip()
    if image_id:
        indexed_path = image_index.get(normalize_image_id(image_id))
        if indexed_path is not None:
            return indexed_path

        fallback = project_root / image_dir / f"{image_id}{image_ext}"
        if fallback.exists():
            return fallback
    return None


def count_distribution(rows, column):
    return dict(Counter(row.get(column, "unknown") or "unknown" for row in rows).most_common())


def validate_image_files(rows, project_root, image_dir, image_ext, max_open, image_index):
    existing = 0
    missing = []
    resolved_paths = []

    for row in rows:
        image_path = resolve_image_path(row, project_root, image_dir, image_ext, image_index)
        if image_path is None:
            missing.append(row.get("image_id", ""))
            continue
        existing += 1
        if len(resolved_paths) < max_open:
            resolved_paths.append(image_path)

    open_result = {
        "enabled": max_open > 0,
        "pil_available": False,
        "checked": 0,
        "failed": [],
        "image_sizes": {},
    }
    if max_open <= 0:
        return existing, missing, open_result

    try:
        from PIL import Image
    except ImportError:
        return existing, missing, open_result

    open_result["pil_available"] = True
    for path in resolved_paths:
        try:
            with Image.open(path) as image:
                size = f"{image.width}x{image.height}"
                open_result["image_sizes"][size] = open_result["image_sizes"].get(size, 0) + 1
                open_result["checked"] += 1
        except Exception as exc:
            open_result["failed"].append({"path": str(path), "error": str(exc)})
    return existing, missing, open_result


def collect_split_image_ids(split_summaries):
    image_ids = set()
    for split in split_summaries:
        for image_id in split["all_image_ids"]:
            normalized = normalize_image_id(image_id)
            if normalized:
                image_ids.add(normalized)
    return image_ids


def check_split(split_name, csv_path, project_root, image_dir, image_ext, max_open, image_index):
    rows, fieldnames = read_csv(csv_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    existing_images, missing_images, open_result = validate_image_files(
        rows,
        project_root,
        image_dir,
        image_ext,
        max_open,
        image_index,
    )

    return {
        "split": split_name,
        "csv_path": str(csv_path),
        "num_rows": len(rows),
        "fieldnames": fieldnames,
        "missing_columns": missing_columns,
        "existing_images": existing_images,
        "missing_images": len(missing_images),
        "missing_image_examples": missing_images[:20],
        "all_image_ids": [row.get("image_id", "") for row in rows],
        "missing_image_ids": missing_images,
        "label_distribution": count_distribution(rows, "label"),
        "sex_distribution": count_distribution(rows, "sex"),
        "age_group_distribution": count_distribution(rows, "age_group"),
        "anatom_site_distribution": count_distribution(rows, "anatom_site_general"),
        "unique_lesion_ids": len({row.get("lesion_id", "") for row in rows if row.get("lesion_id", "")}),
        "image_open_check": open_result,
    }


def write_markdown(path, summary):
    lines = [
        "# ISIC 2019 数据可用性检查",
        "",
        "## 一、总体结果",
        "",
        f"- splits_dir：`{summary['splits_dir']}`",
        f"- image_dir：`{summary['image_dir']}`",
        f"- image_dir 是否存在：{summary['image_root_exists']}",
        f"- 递归索引图片数：{summary['indexed_image_files']}",
        f"- 总样本数：{summary['total_rows']}",
        f"- 图片存在数：{summary['total_existing_images']}",
        f"- 图片缺失数：{summary['total_missing_images']}",
        f"- split 中不在图片目录的 ID 数：{summary['split_ids_missing_in_images_count']}",
        f"- 图片目录中不在 split 的 ID 数：{summary['image_ids_not_used_by_split_count']}",
        "",
    ]

    if summary["split_ids_missing_in_images_examples"]:
        lines += [
            "## 二、ID 差异示例",
            "",
            "### split 中存在但图片目录不存在",
            "",
            "```text",
        ]
        lines.extend(summary["split_ids_missing_in_images_examples"])
        lines += ["```", ""]

    if summary["image_ids_not_used_by_split_examples"]:
        lines += [
            "### 图片目录存在但 split 中没有使用",
            "",
            "```text",
        ]
        lines.extend(summary["image_ids_not_used_by_split_examples"])
        lines += ["```", ""]

    for split in summary["splits"]:
        lines += [
            f"## {split['split']}",
            "",
            f"- 样本数：{split['num_rows']}",
            f"- 图片存在数：{split['existing_images']}",
            f"- 图片缺失数：{split['missing_images']}",
            f"- lesion_id 数：{split['unique_lesion_ids']}",
            f"- 缺失字段：{split['missing_columns']}",
            "",
            "### 标签分布",
            "",
            "| 标签 | 数量 |",
            "|---|---:|",
        ]
        for key, value in split["label_distribution"].items():
            lines.append(f"| {key} | {value} |")

        lines += ["", "### 性别分布", "", "| 性别 | 数量 |", "|---|---:|"]
        for key, value in split["sex_distribution"].items():
            lines.append(f"| {key} | {value} |")

        lines += ["", "### 年龄段分布", "", "| 年龄段 | 数量 |", "|---|---:|"]
        for key, value in split["age_group_distribution"].items():
            lines.append(f"| {key} | {value} |")

        lines += ["", "### 解剖部位分布", "", "| 部位 | 数量 |", "|---|---:|"]
        for key, value in split["anatom_site_distribution"].items():
            lines.append(f"| {key} | {value} |")

        open_check = split["image_open_check"]
        lines += [
            "",
            "### 图片抽样打开检查",
            "",
            f"- 启用：{open_check['enabled']}",
            f"- PIL 可用：{open_check['pil_available']}",
            f"- 成功打开数量：{open_check['checked']}",
            f"- 打开失败数量：{len(open_check['failed'])}",
            "",
        ]
        if split["missing_image_examples"]:
            lines += ["### 缺失图片示例", "", "```text"]
            lines.extend(split["missing_image_examples"])
            lines += ["```", ""]

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Check ISIC 2019 split CSV and image availability.")
    parser.add_argument("--splits-dir", type=str, default="week6/experiments/isic2019_splits")
    parser.add_argument("--image-dir", type=str, default="data/isic2019/images")
    parser.add_argument("--image-ext", type=str, default=".jpg")
    parser.add_argument("--output-dir", type=str, default="week7/experiments/stage01_isic2019_data_check")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--max-open-per-split", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    splits_dir = project_root / args.splits_dir
    image_dir = Path(args.image_dir)
    output_dir = project_root / args.output_dir
    image_root = project_root / image_dir

    if not splits_dir.exists():
        raise FileNotFoundError(f"split 目录不存在：{splits_dir}")
    image_index = build_image_index(image_root)

    split_summaries = []
    for split_name in ["train", "val", "test"]:
        csv_path = splits_dir / f"{split_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"split CSV 不存在：{csv_path}")
        split_summaries.append(
            check_split(
                split_name,
                csv_path,
                project_root,
                image_dir,
                args.image_ext,
                args.max_open_per_split,
                image_index,
            )
        )

    split_image_ids = collect_split_image_ids(split_summaries)
    indexed_image_ids = set(image_index)
    split_ids_missing_in_images = sorted(split_image_ids - indexed_image_ids)
    image_ids_not_used_by_split = sorted(indexed_image_ids - split_image_ids)

    for split in split_summaries:
        split.pop("all_image_ids", None)

    summary = {
        "splits_dir": str(splits_dir),
        "image_dir": str(image_root),
        "image_root_exists": image_root.exists(),
        "indexed_image_files": len(image_index),
        "total_rows": sum(split["num_rows"] for split in split_summaries),
        "total_existing_images": sum(split["existing_images"] for split in split_summaries),
        "total_missing_images": sum(split["missing_images"] for split in split_summaries),
        "split_ids_missing_in_images_count": len(split_ids_missing_in_images),
        "split_ids_missing_in_images_examples": split_ids_missing_in_images[:50],
        "image_ids_not_used_by_split_count": len(image_ids_not_used_by_split),
        "image_ids_not_used_by_split_examples": image_ids_not_used_by_split[:50],
        "splits": split_summaries,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "data_check_summary.json"
    md_path = output_dir / "data_check_summary.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, summary)
    (output_dir / "split_ids_missing_in_images.txt").write_text(
        "\n".join(split_ids_missing_in_images) + ("\n" if split_ids_missing_in_images else ""),
        encoding="utf-8",
    )
    (output_dir / "image_ids_not_used_by_split.txt").write_text(
        "\n".join(image_ids_not_used_by_split) + ("\n" if image_ids_not_used_by_split else ""),
        encoding="utf-8",
    )

    print(f"JSON 结果已保存：{json_path}")
    print(f"Markdown 结果已保存：{md_path}")
    if summary["total_missing_images"] > 0:
        print(f"警告：存在缺失图片 {summary['total_missing_images']} 张")


if __name__ == "__main__":
    main()
