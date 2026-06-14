import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LABELS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
GROUP_COLUMNS = ["label", "sex", "age_group", "anatom_site_general"]


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_log(log_path, message):
    print(message)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def read_rows(csv_path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_image_index(image_root):
    image_root = Path(image_root)
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


def resolve_image_path(row, project_root, image_index):
    image_path = row.get("image_path", "").strip()
    if image_path:
        path = Path(image_path)
        if not path.is_absolute():
            path = project_root / path
        if path.exists():
            return path

    image_id = row.get("image_id", "").strip()
    indexed_path = image_index.get(image_id.lower())
    if indexed_path is not None:
        return indexed_path
    raise FileNotFoundError(f"图片不存在：{image_id}")


class Isic2019Dataset(Dataset):
    def __init__(self, csv_path, image_root, transform=None):
        self.csv_path = Path(csv_path)
        self.project_root = PROJECT_ROOT
        self.transform = transform
        self.label_to_idx = {label: idx for idx, label in enumerate(LABELS)}
        self.image_index = build_image_index(image_root)
        raw_rows = read_rows(self.csv_path)
        self.rows = []

        for row in raw_rows:
            label = row.get("label", "")
            if label not in self.label_to_idx:
                continue
            image_path = resolve_image_path(row, self.project_root, self.image_index)
            item = dict(row)
            item["resolved_image_path"] = str(image_path)
            item["label_idx"] = self.label_to_idx[label]
            self.rows.append(item)

        if not self.rows:
            raise ValueError(f"没有可训练样本：{csv_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["resolved_image_path"]) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        label = torch.tensor(row["label_idx"], dtype=torch.long)
        groups = {column: row.get(column, "unknown") or "unknown" for column in GROUP_COLUMNS}
        return image, label, groups


def collate_batch(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.stack([item[1] for item in batch], dim=0)
    groups = {
        column: [item[2][column] for item in batch]
        for column in GROUP_COLUMNS
    }
    return images, labels, groups


def build_transforms(image_size):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def build_subset(dataset, size, seed):
    if size <= 0 or size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def get_dataset_rows(dataset):
    if isinstance(dataset, Subset):
        return [dataset.dataset.rows[index] for index in dataset.indices]
    return dataset.rows


def build_model(num_classes, pretrained):
    try:
        from torchvision.models import ResNet18_Weights

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
    except Exception:
        model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def compute_class_weights(dataset, num_classes, device):
    rows = get_dataset_rows(dataset)
    counts = Counter(row["label_idx"] for row in rows)
    total = sum(counts.values())
    weights = []
    for class_idx in range(num_classes):
        count = counts.get(class_idx, 0)
        weight = total / (num_classes * count) if count > 0 else 0.0
        weights.append(weight)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return weight_tensor / weight_tensor.mean().clamp_min(1e-8)


def build_train_sampler(dataset, sampler_mode, seed):
    if sampler_mode == "none":
        return None

    rows = get_dataset_rows(dataset)
    if sampler_mode == "balanced_label":
        keys = [row["label"] for row in rows]
    elif sampler_mode == "balanced_anatom_site":
        keys = [row.get("anatom_site_general", "unknown") or "unknown" for row in rows]
    else:
        raise ValueError(f"未知采样模式：{sampler_mode}")

    counts = Counter(keys)
    sample_weights = [1.0 / counts[key] for key in keys]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


def update_group_stats(group_stats, groups, preds, labels):
    preds_cpu = preds.detach().cpu().tolist()
    labels_cpu = labels.detach().cpu().tolist()
    for column, values in groups.items():
        for value, pred, label in zip(values, preds_cpu, labels_cpu):
            stats = group_stats[column][value]
            stats["seen"] += 1
            stats["correct"] += int(pred == label)


def summarize_group_stats(group_stats):
    result = {}
    for column, values in group_stats.items():
        column_result = {}
        accuracies = []
        for value, stats in values.items():
            seen = stats["seen"]
            accuracy = stats["correct"] / seen if seen > 0 else 0.0
            column_result[value] = {
                "seen": seen,
                "correct": stats["correct"],
                "accuracy": accuracy,
            }
            accuracies.append(accuracy)
        result[column] = {
            "groups": column_result,
            "worst_group_accuracy": min(accuracies) if accuracies else 0.0,
            "macro_group_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
        }
    return result


def evaluate(model, dataloader, criterion, device, num_classes, use_amp):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    class_correct = [0 for _ in range(num_classes)]
    class_seen = [0 for _ in range(num_classes)]
    group_stats = {column: defaultdict(lambda: {"seen": 0, "correct": 0}) for column in GROUP_COLUMNS}

    with torch.no_grad():
        for images, labels, groups in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_seen += labels.size(0)

            for class_idx in range(num_classes):
                mask = labels == class_idx
                seen = int(mask.sum().item())
                if seen > 0:
                    class_seen[class_idx] += seen
                    class_correct[class_idx] += int((preds[mask] == labels[mask]).sum().item())
            update_group_stats(group_stats, groups, preds, labels)

    class_accuracy = [
        class_correct[idx] / class_seen[idx] if class_seen[idx] > 0 else 0.0
        for idx in range(num_classes)
    ]
    seen_class_accuracy = [
        class_accuracy[idx]
        for idx in range(num_classes)
        if class_seen[idx] > 0
    ]
    return {
        "loss": total_loss / total_seen,
        "accuracy": total_correct / total_seen,
        "balanced_accuracy": sum(seen_class_accuracy) / len(seen_class_accuracy),
        "class_accuracy": {LABELS[idx]: class_accuracy[idx] for idx in range(num_classes)},
        "class_seen": {LABELS[idx]: class_seen[idx] for idx in range(num_classes)},
        "worst_class_accuracy": min(seen_class_accuracy) if seen_class_accuracy else 0.0,
        "group_metrics": summarize_group_stats(group_stats),
    }


def train_one_epoch(model, dataloader, criterion, optimizer, device, use_amp, scaler):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, labels, _ in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_seen += labels.size(0)

    return {
        "loss": total_loss / total_seen,
        "accuracy": total_correct / total_seen,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="ISIC 2019 clean baseline training.")
    parser.add_argument("--splits-dir", type=str, default="week7/experiments/isic2019_splits_available")
    parser.add_argument("--image-dir", type=str, default="data/isic2019/images")
    parser.add_argument("--output-dir", type=str, default="week7/experiments/stage02_isic2019_baseline_smoke")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight", type=str, default="balanced", choices=["none", "balanced"])
    parser.add_argument(
        "--sampler",
        type=str,
        default="none",
        choices=["none", "balanced_label", "balanced_anatom_site"],
    )
    parser.add_argument("--train-size", type=int, default=0)
    parser.add_argument("--val-size", type=int, default=0)
    parser.add_argument("--test-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.txt"
    if log_path.exists():
        log_path.unlink()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        if args.disable_cudnn:
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    use_amp = args.amp and device.type == "cuda"

    train_transform, eval_transform = build_transforms(args.image_size)
    splits_dir = Path(args.splits_dir)
    train_dataset = Isic2019Dataset(splits_dir / "train.csv", args.image_dir, transform=train_transform)
    val_dataset = Isic2019Dataset(splits_dir / "val.csv", args.image_dir, transform=eval_transform)
    test_dataset = Isic2019Dataset(splits_dir / "test.csv", args.image_dir, transform=eval_transform)
    train_dataset = build_subset(train_dataset, args.train_size, args.seed)
    val_dataset = build_subset(val_dataset, args.val_size, args.seed + 1)
    test_dataset = build_subset(test_dataset, args.test_size, args.seed + 2)

    persistent_workers = args.num_workers > 0
    train_sampler = build_train_sampler(train_dataset, args.sampler, args.seed)
    trainloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_batch,
    )
    valloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_batch,
    )
    testloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_batch,
    )

    model = build_model(len(LABELS), args.pretrained).to(device)
    class_weights = compute_class_weights(train_dataset, len(LABELS), device) if args.class_weight == "balanced" else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    config = vars(args)
    config["device"] = str(device)
    config["num_classes"] = len(LABELS)
    config["labels"] = LABELS
    write_log(log_path, "实验配置：" + json.dumps(config, ensure_ascii=False, indent=2))

    history = []
    best_val_balanced = -1.0
    best_model_path = output_dir / "model_best.pth"
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, trainloader, criterion, optimizer, device, use_amp, scaler)
        val_metrics = evaluate(model, valloader, criterion, device, len(LABELS), use_amp)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_worst_class_accuracy": val_metrics["worst_class_accuracy"],
        }
        history.append(epoch_metrics)
        torch.save(model.state_dict(), output_dir / "model_latest.pth")
        if val_metrics["balanced_accuracy"] > best_val_balanced:
            best_val_balanced = val_metrics["balanced_accuracy"]
            torch.save(model.state_dict(), best_model_path)

        with (output_dir / "history_latest.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        write_log(
            log_path,
            (
                f"epoch {epoch}/{args.epochs} | train loss: {train_metrics['loss']:.4f} | "
                f"train acc: {train_metrics['accuracy']:.4f} | val acc: {val_metrics['accuracy']:.4f} | "
                f"val balanced acc: {val_metrics['balanced_accuracy']:.4f} | "
                f"val worst-class acc: {val_metrics['worst_class_accuracy']:.4f}"
            ),
        )

    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    val_metrics = evaluate(model, valloader, criterion, device, len(LABELS), use_amp)
    test_metrics = evaluate(model, testloader, criterion, device, len(LABELS), use_amp)
    metrics = {
        "config": config,
        "history": history,
        "val_eval": val_metrics,
        "test_eval": test_metrics,
        "method_note": "ISIC 2019 clean baseline，按疾病类别、性别、年龄段和解剖部位输出分组评测。",
    }
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    torch.save(model.state_dict(), output_dir / "model.pth")

    write_log(log_path, "最终 val accuracy: {:.4f}".format(val_metrics["accuracy"]))
    write_log(log_path, "最终 val balanced accuracy: {:.4f}".format(val_metrics["balanced_accuracy"]))
    write_log(log_path, "最终 val worst-class accuracy: {:.4f}".format(val_metrics["worst_class_accuracy"]))
    write_log(log_path, "最终 test accuracy: {:.4f}".format(test_metrics["accuracy"]))
    write_log(log_path, "最终 test balanced accuracy: {:.4f}".format(test_metrics["balanced_accuracy"]))
    write_log(log_path, "最终 test worst-class accuracy: {:.4f}".format(test_metrics["worst_class_accuracy"]))
    write_log(log_path, f"指标已保存：{metrics_path}")
    write_log(log_path, f"模型已保存：{output_dir / 'model.pth'}")


if __name__ == "__main__":
    main()
