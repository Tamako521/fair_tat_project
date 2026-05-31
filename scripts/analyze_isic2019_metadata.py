import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DISEASE_COLUMNS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
AGE_BINS = [
    ("0-30", 0, 30),
    ("31-45", 31, 45),
    ("46-60", 46, 60),
    ("61+", 61, 200),
]


def normalize_key(row, candidates):
    for key in candidates:
        if key in row:
            return key
    return None


def is_missing(value):
    if value is None:
        return True
    value = str(value).strip()
    return value == "" or value.lower() in {"nan", "none", "null", "unknown", "unk"}


def get_label(row):
    for column in DISEASE_COLUMNS:
        value = str(row.get(column, "")).strip()
        if value in {"1", "1.0", "true", "True"}:
            return column

    for column in ["diagnosis", "label", "target", "benign_malignant"]:
        value = row.get(column)
        if not is_missing(value):
            return str(value).strip()

    return "unknown"


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


def count_values(rows, key, transform=None):
    counter = Counter()
    if key is None:
        counter["missing_column"] = len(rows)
        return counter

    for row in rows:
        value = row.get(key)
        if transform is not None:
            value = transform(value)
        elif is_missing(value):
            value = "unknown"
        else:
            value = str(value).strip().lower()
        counter[value] += 1
    return counter


def count_missing(rows, fieldnames):
    result = {}
    for field in fieldnames:
        result[field] = sum(1 for row in rows if is_missing(row.get(field)))
    return result


def count_duplicates(rows, key):
    if key is None:
        return {"available": False, "unique": 0, "duplicated_ids": 0, "max_images_per_id": 0}
    counter = Counter(row.get(key) for row in rows if not is_missing(row.get(key)))
    duplicated = sum(1 for _, count in counter.items() if count > 1)
    max_count = max(counter.values(), default=0)
    return {
        "available": True,
        "unique": len(counter),
        "duplicated_ids": duplicated,
        "max_images_per_id": max_count,
    }


def counter_to_dict(counter):
    return dict(counter.most_common())


def write_markdown(output_path, summary):
    lines = [
        "# ISIC 2019 Metadata 统计结果",
        "",
        "## 一、基本信息",
        "",
        f"- 样本数：{summary['num_rows']}",
        f"- 字段数：{len(summary['fieldnames'])}",
        "",
        "## 二、疾病类别分布",
        "",
        "| 类别 | 数量 |",
        "|---|---:|",
    ]
    for key, value in summary["label_distribution"].items():
        lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "## 三、性别分布",
        "",
        "| 性别 | 数量 |",
        "|---|---:|",
    ]
    for key, value in summary["sex_distribution"].items():
        lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "## 四、年龄段分布",
        "",
        "| 年龄段 | 数量 |",
        "|---|---:|",
    ]
    for key, value in summary["age_group_distribution"].items():
        lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "## 五、解剖部位分布",
        "",
        "| 解剖部位 | 数量 |",
        "|---|---:|",
    ]
    for key, value in summary["anatom_site_distribution"].items():
        lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "## 六、患者与病灶重复情况",
        "",
        f"- patient_id 可用：{summary['patient_id_stats']['available']}",
        f"- patient_id 唯一数量：{summary['patient_id_stats']['unique']}",
        f"- 出现多张图像的 patient_id 数：{summary['patient_id_stats']['duplicated_ids']}",
        f"- 单个 patient_id 最大图像数：{summary['patient_id_stats']['max_images_per_id']}",
        f"- lesion_id 可用：{summary['lesion_id_stats']['available']}",
        f"- lesion_id 唯一数量：{summary['lesion_id_stats']['unique']}",
        f"- 出现多张图像的 lesion_id 数：{summary['lesion_id_stats']['duplicated_ids']}",
        f"- 单个 lesion_id 最大图像数：{summary['lesion_id_stats']['max_images_per_id']}",
        "",
        "## 七、缺失值最多的字段",
        "",
        "| 字段 | 缺失数量 |",
        "|---|---:|",
    ]
    for key, value in list(summary["missing_values"].items())[:20]:
        lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "## 八、后续建议",
        "",
        "1. 优先使用 patient-level split，避免同一患者进入训练集和验证集。",
        "2. 训练和评测时同时关注疾病类别、性别、年龄段和解剖部位分组。",
        "3. 缺失值较多的属性应保留 `unknown` 分组，不建议直接删除样本。",
    ]

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_metadata(metadata_path, output_dir):
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    sex_key = normalize_key(rows[0] if rows else {}, ["sex", "gender"])
    age_key = normalize_key(rows[0] if rows else {}, ["age_approx", "age", "approx_age"])
    anatom_key = normalize_key(
        rows[0] if rows else {},
        ["anatom_site_general", "anatom_site_general_challenge", "anatom_site", "localization"],
    )
    patient_key = normalize_key(rows[0] if rows else {}, ["patient_id", "patient", "patient_code"])
    lesion_key = normalize_key(rows[0] if rows else {}, ["lesion_id", "lesion"])

    label_distribution = Counter(get_label(row) for row in rows)
    missing_values = count_missing(rows, fieldnames)
    missing_values = dict(sorted(missing_values.items(), key=lambda item: item[1], reverse=True))

    summary = {
        "metadata_path": str(metadata_path),
        "num_rows": len(rows),
        "fieldnames": fieldnames,
        "detected_columns": {
            "sex": sex_key,
            "age": age_key,
            "anatom_site": anatom_key,
            "patient_id": patient_key,
            "lesion_id": lesion_key,
        },
        "label_distribution": counter_to_dict(label_distribution),
        "sex_distribution": counter_to_dict(count_values(rows, sex_key)),
        "age_group_distribution": counter_to_dict(count_values(rows, age_key, transform=get_age_group)),
        "anatom_site_distribution": counter_to_dict(count_values(rows, anatom_key)),
        "patient_id_stats": count_duplicates(rows, patient_key),
        "lesion_id_stats": count_duplicates(rows, lesion_key),
        "missing_values": missing_values,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "isic2019_metadata_summary.json"
    md_path = output_dir / "isic2019_metadata_summary.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, summary)
    return json_path, md_path


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze ISIC 2019 metadata for fairness evaluation.")
    parser.add_argument("--metadata", type=str, required=True, help="Path to ISIC 2019 metadata CSV.")
    parser.add_argument("--output-dir", type=str, default="week6/experiments/isic2019_metadata_analysis")
    return parser.parse_args()


def main():
    args = parse_args()
    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata 文件不存在：{metadata_path}")

    json_path, md_path = analyze_metadata(metadata_path, output_dir)
    print(f"JSON 结果已保存：{json_path}")
    print(f"Markdown 结果已保存：{md_path}")


if __name__ == "__main__":
    main()
