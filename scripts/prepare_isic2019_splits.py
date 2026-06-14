import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


DISEASE_LABELS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
AGE_BINS = [
    ("0-30", 0, 30),
    ("31-45", 31, 45),
    ("46-60", 46, 60),
    ("61+", 61, 200),
]


def is_missing(value):
    if value is None:
        return True
    value = str(value).strip()
    return value == "" or value.lower() in {"nan", "none", "null", "unknown", "unk"}


def normalize_label(label):
    label = str(label).strip().lower()
    mapping = {
        "melanoma": "MEL",
        "nevus": "NV",
        "basal cell carcinoma": "BCC",
        "actinic keratosis": "AK",
        "pigmented benign keratosis": "BKL",
        "seborrheic keratosis": "BKL",
        "solar lentigo": "BKL",
        "dermatofibroma": "DF",
        "vascular lesion": "VASC",
        "squamous cell carcinoma": "SCC",
    }
    return mapping.get(label, label.upper())


def get_label(row):
    for column in DISEASE_LABELS:
        value = str(row.get(column, "")).strip()
        if value in {"1", "1.0", "true", "True"}:
            return column
    for column in ["diagnosis", "label", "target"]:
        if not is_missing(row.get(column)):
            return normalize_label(row[column])
    return "unknown"


def get_image_id(row):
    for column in ["image", "image_id", "isic_id", "name"]:
        if not is_missing(row.get(column)):
            return str(row[column]).strip()
    return ""


def get_age_group(value):
    if is_missing(value):
        return "unknown"
    try:
        age = float(value)
    except ValueError:
        return "unknown"
    for name, start, end in AGE_BINS:
        if start <= age <= end:
            return name
    return "unknown"


def read_rows(csv_path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def merge_groundtruth(rows, groundtruth_path):
    groundtruth_rows = read_rows(groundtruth_path)
    if not rows or not groundtruth_rows:
        return rows, 0

    metadata_by_image = {}
    for row in rows:
        image_id = get_image_id(row)
        if image_id:
            metadata_by_image[image_id.lower()] = row

    merged_count = 0
    for gt_row in groundtruth_rows:
        image_id = get_image_id(gt_row)
        if not image_id:
            continue
        metadata_row = metadata_by_image.get(image_id.lower())
        if metadata_row is None:
            continue
        for label in DISEASE_LABELS:
            if label in gt_row:
                metadata_row[label] = gt_row[label]
        merged_count += 1
    return rows, merged_count


def build_image_path(image_dir, image_id, image_ext):
    if not image_id:
        return ""
    return str(Path(image_dir) / f"{image_id}{image_ext}")


def build_image_index(image_dir):
    image_root = Path(image_dir)
    if not image_root.exists():
        return {}

    image_index = {}
    for path in image_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue
        stem = path.stem.lower()
        image_index.setdefault(stem, path)
        if stem.endswith("_downsampled"):
            image_index.setdefault(stem.removesuffix("_downsampled"), path)
    return image_index


def normalize_row(row, image_dir, image_ext, image_index=None):
    image_id = get_image_id(row)
    label = get_label(row)
    sex = row.get("sex", "unknown")
    age = row.get("age_approx", row.get("age", ""))
    anatom_site = row.get("anatom_site_general", "unknown")
    lesion_id = row.get("lesion_id", "")
    patient_id = row.get("patient_id", "")
    indexed_path = None
    if image_index is not None and image_id:
        indexed_path = image_index.get(image_id.lower())

    return {
        "image_id": image_id,
        "image_path": str(indexed_path) if indexed_path is not None else build_image_path(image_dir, image_id, image_ext),
        "label": label,
        "sex": "unknown" if is_missing(sex) else str(sex).strip().lower(),
        "age_approx": "" if is_missing(age) else str(age).strip(),
        "age_group": get_age_group(age),
        "anatom_site_general": "unknown" if is_missing(anatom_site) else str(anatom_site).strip().lower(),
        "lesion_id": "" if is_missing(lesion_id) else str(lesion_id).strip(),
        "patient_id": "" if is_missing(patient_id) else str(patient_id).strip(),
    }


def split_groups(rows, train_ratio, val_ratio, seed):
    grouped = defaultdict(list)
    for row in rows:
        group_key = row["lesion_id"] if row["lesion_id"] else row["image_id"]
        grouped[group_key].append(row)

    groups_by_label = defaultdict(list)
    for group_rows in grouped.values():
        label = Counter(row["label"] for row in group_rows).most_common(1)[0][0]
        groups_by_label[label].append(group_rows)

    rng = random.Random(seed)
    split_rows = {"train": [], "val": [], "test": []}
    for label, label_groups in groups_by_label.items():
        rng.shuffle(label_groups)
        total_rows = sum(len(group) for group in label_groups)
        train_target = total_rows * train_ratio
        val_target = total_rows * val_ratio
        train_count = 0
        val_count = 0

        for group in label_groups:
            if train_count < train_target:
                split_name = "train"
                train_count += len(group)
            elif val_count < val_target:
                split_name = "val"
                val_count += len(group)
            else:
                split_name = "test"
            for row in group:
                row_with_split = dict(row)
                row_with_split["split"] = split_name
                split_rows[split_name].append(row_with_split)

    return split_rows


def summarize(split_rows):
    summary = {}
    for split_name, rows in split_rows.items():
        summary[split_name] = {
            "num_rows": len(rows),
            "label_distribution": dict(Counter(row["label"] for row in rows).most_common()),
            "sex_distribution": dict(Counter(row["sex"] for row in rows).most_common()),
            "age_group_distribution": dict(Counter(row["age_group"] for row in rows).most_common()),
            "anatom_site_distribution": dict(Counter(row["anatom_site_general"] for row in rows).most_common()),
            "unique_lesion_ids": len({row["lesion_id"] for row in rows if row["lesion_id"]}),
        }
    return summary


def write_csv(path, rows):
    fieldnames = [
        "split",
        "image_id",
        "image_path",
        "label",
        "sex",
        "age_approx",
        "age_group",
        "anatom_site_general",
        "lesion_id",
        "patient_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, summary):
    lines = ["# ISIC 2019 Split 统计结果", ""]
    for split_name, stats in summary.items():
        lines += [
            f"## {split_name}",
            "",
            f"- 样本数：{stats['num_rows']}",
            f"- lesion_id 数：{stats['unique_lesion_ids']}",
            "",
            "### 标签分布",
            "",
            "| 标签 | 数量 |",
            "|---|---:|",
        ]
        for key, value in stats["label_distribution"].items():
            lines.append(f"| {key} | {value} |")
        lines += ["", "### 性别分布", "", "| 性别 | 数量 |", "|---|---:|"]
        for key, value in stats["sex_distribution"].items():
            lines.append(f"| {key} | {value} |")
        lines += ["", "### 年龄段分布", "", "| 年龄段 | 数量 |", "|---|---:|"]
        for key, value in stats["age_group_distribution"].items():
            lines.append(f"| {key} | {value} |")
        lines += ["", "### 解剖部位分布", "", "| 部位 | 数量 |", "|---|---:|"]
        for key, value in stats["anatom_site_distribution"].items():
            lines.append(f"| {key} | {value} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare ISIC 2019 metadata splits.")
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--groundtruth", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="week6/experiments/isic2019_splits")
    parser.add_argument("--image-dir", type=str, default="data/isic2019/images")
    parser.add_argument("--image-ext", type=str, default=".jpg")
    parser.add_argument("--require-images", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata 文件不存在：{metadata_path}")
    groundtruth_path = Path(args.groundtruth) if args.groundtruth is not None else None
    if groundtruth_path is not None and not groundtruth_path.exists():
        raise FileNotFoundError(f"ground truth 文件不存在：{groundtruth_path}")

    image_index = build_image_index(args.image_dir) if args.require_images else None
    if args.require_images and not image_index:
        raise FileNotFoundError(f"没有在图片目录中找到图片：{args.image_dir}")

    raw_rows = read_rows(metadata_path)
    if groundtruth_path is not None:
        raw_rows, merged_count = merge_groundtruth(raw_rows, groundtruth_path)
        print(f"ground truth 合并数量：{merged_count}")

    rows = [
        normalize_row(row, args.image_dir, args.image_ext, image_index=image_index)
        for row in raw_rows
    ]
    total_normalized_rows = len(rows)
    rows = [row for row in rows if row["image_id"] and row["label"] in DISEASE_LABELS]
    labeled_rows = len(rows)
    if args.require_images:
        rows = [row for row in rows if row["image_id"].lower() in image_index]
    if not rows:
        raise ValueError(
            "没有可用于划分的数据。"
            f"原始行数={total_normalized_rows}, 有效标签行数={labeled_rows}, "
            f"图片索引数={len(image_index) if image_index is not None else '未启用'}。"
            "请检查是否需要传入 --groundtruth，或 image_id 是否与图片文件名匹配。"
        )

    split_rows = split_groups(rows, args.train_ratio, args.val_ratio, args.seed)
    all_rows = split_rows["train"] + split_rows["val"] + split_rows["test"]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "isic2019_splits.csv", all_rows)
    write_csv(output_dir / "train.csv", split_rows["train"])
    write_csv(output_dir / "val.csv", split_rows["val"])
    write_csv(output_dir / "test.csv", split_rows["test"])

    summary = summarize(split_rows)
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(output_dir / "split_summary.md", summary)
    print(f"划分文件已保存：{output_dir}")


if __name__ == "__main__":
    main()
