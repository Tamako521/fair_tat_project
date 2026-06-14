import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_isic2019_baseline import (
    GROUP_COLUMNS,
    LABELS,
    Isic2019Dataset,
    build_model,
    build_subset,
    build_transforms,
    collate_batch,
    summarize_group_stats,
    update_group_stats,
)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def write_log(log_path, message):
    print(message)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def build_normalized_bounds(device):
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std
    return mean, std, lower, upper


def clamp_normalized(images, lower, upper):
    return torch.max(torch.min(images, upper), lower)


def project_linf_normalized(adv_images, clean_images, epsilon, std, lower, upper):
    epsilon_tensor = torch.tensor(epsilon, device=adv_images.device).view(1, 1, 1, 1) / std
    delta = torch.clamp(adv_images - clean_images, min=-epsilon_tensor, max=epsilon_tensor)
    return clamp_normalized(clean_images + delta, lower, upper)


def fgsm_attack(model, images, labels, criterion, epsilon, std, lower, upper, use_amp=False):
    adv_images = images.detach().clone().requires_grad_(True)
    with torch.cuda.amp.autocast(enabled=use_amp):
        outputs = model(adv_images)
        loss = criterion(outputs, labels)
    model.zero_grad(set_to_none=True)
    loss.backward()
    step = torch.tensor(epsilon, device=images.device).view(1, 1, 1, 1) / std
    adv_images = adv_images + step * adv_images.grad.sign()
    return project_linf_normalized(adv_images.detach(), images, epsilon, std, lower, upper)


def pgd_attack(model, images, labels, criterion, epsilon, alpha, steps, std, lower, upper, use_amp=False):
    clean_images = images.detach()
    noise = torch.empty_like(clean_images).uniform_(-1.0, 1.0)
    epsilon_tensor = torch.tensor(epsilon, device=images.device).view(1, 1, 1, 1) / std
    adv_images = clamp_normalized(clean_images + noise * epsilon_tensor, lower, upper)

    for _ in range(steps):
        adv_images = adv_images.detach().requires_grad_(True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(adv_images)
            loss = criterion(outputs, labels)
        model.zero_grad(set_to_none=True)
        loss.backward()
        alpha_tensor = torch.tensor(alpha, device=images.device).view(1, 1, 1, 1) / std
        adv_images = adv_images + alpha_tensor * adv_images.grad.sign()
        adv_images = project_linf_normalized(adv_images.detach(), clean_images, epsilon, std, lower, upper)
    return adv_images.detach()


def random_noise_attack(images, epsilon, std, lower, upper):
    noise = torch.empty_like(images).uniform_(-1.0, 1.0)
    epsilon_tensor = torch.tensor(epsilon, device=images.device).view(1, 1, 1, 1) / std
    return clamp_normalized(images + noise * epsilon_tensor, lower, upper)


def evaluate_attack(model, dataloader, criterion, device, args):
    model.eval()
    _, std, lower, upper = build_normalized_bounds(device)
    total_seen = 0
    clean_correct = 0
    adv_correct = 0
    class_seen = [0 for _ in LABELS]
    clean_class_correct = [0 for _ in LABELS]
    adv_class_correct = [0 for _ in LABELS]
    clean_group_stats = {column: defaultdict(lambda: {"seen": 0, "correct": 0}) for column in GROUP_COLUMNS}
    adv_group_stats = {column: defaultdict(lambda: {"seen": 0, "correct": 0}) for column in GROUP_COLUMNS}

    for images, labels, groups in dataloader:
        if total_seen >= args.eval_size:
            break
        if total_seen + labels.size(0) > args.eval_size:
            keep = args.eval_size - total_seen
            images = images[:keep]
            labels = labels[:keep]
            groups = {column: values[:keep] for column, values in groups.items()}

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                clean_outputs = model(images)
            clean_preds = clean_outputs.argmax(dim=1)

        if args.attack == "fgsm":
            adv_images = fgsm_attack(
                model,
                images,
                labels,
                criterion,
                args.epsilon,
                std,
                lower,
                upper,
                use_amp=args.amp and device.type == "cuda",
            )
        elif args.attack == "pgd":
            adv_images = pgd_attack(
                model,
                images,
                labels,
                criterion,
                args.epsilon,
                args.alpha,
                args.steps,
                std,
                lower,
                upper,
                use_amp=args.amp and device.type == "cuda",
            )
        elif args.attack == "random":
            adv_images = random_noise_attack(images, args.epsilon, std, lower, upper)
        else:
            raise ValueError(f"未知攻击方式：{args.attack}")

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                adv_outputs = model(adv_images)
            adv_preds = adv_outputs.argmax(dim=1)

        clean_correct += (clean_preds == labels).sum().item()
        adv_correct += (adv_preds == labels).sum().item()
        total_seen += labels.size(0)

        for class_idx in range(len(LABELS)):
            mask = labels == class_idx
            seen = int(mask.sum().item())
            if seen > 0:
                class_seen[class_idx] += seen
                clean_class_correct[class_idx] += int((clean_preds[mask] == labels[mask]).sum().item())
                adv_class_correct[class_idx] += int((adv_preds[mask] == labels[mask]).sum().item())
        update_group_stats(clean_group_stats, groups, clean_preds, labels)
        update_group_stats(adv_group_stats, groups, adv_preds, labels)

    clean_class_accuracy = {}
    adv_class_accuracy = {}
    seen_adv_rates = []
    for idx, label in enumerate(LABELS):
        clean_rate = clean_class_correct[idx] / class_seen[idx] if class_seen[idx] > 0 else 0.0
        adv_rate = adv_class_correct[idx] / class_seen[idx] if class_seen[idx] > 0 else 0.0
        clean_class_accuracy[label] = clean_rate
        adv_class_accuracy[label] = adv_rate
        if class_seen[idx] > 0:
            seen_adv_rates.append(adv_rate)

    return {
        "attack": args.attack,
        "eval_size": total_seen,
        "epsilon": args.epsilon,
        "alpha": args.alpha if args.attack == "pgd" else None,
        "steps": args.steps if args.attack == "pgd" else None,
        "clean_accuracy": clean_correct / total_seen,
        "adversarial_accuracy": adv_correct / total_seen,
        "adversarial_worst_class_accuracy": min(seen_adv_rates) if seen_adv_rates else 0.0,
        "class_seen": {LABELS[idx]: class_seen[idx] for idx in range(len(LABELS))},
        "clean_class_accuracy": clean_class_accuracy,
        "adversarial_class_accuracy": adv_class_accuracy,
        "clean_group_metrics": summarize_group_stats(clean_group_stats),
        "adversarial_group_metrics": summarize_group_stats(adv_group_stats),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="ISIC 2019 adversarial evaluation.")
    parser.add_argument("--splits-dir", type=str, default="week7/experiments/isic2019_splits_available")
    parser.add_argument("--image-dir", type=str, default="data/isic2019/images")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="week8/experiments/stage01_isic2019_fgsm_eval")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--attack", type=str, default="fgsm", choices=["fgsm", "pgd", "random"])
    parser.add_argument("--epsilon", type=float, default=2 / 255)
    parser.add_argument("--alpha", type=float, default=1 / 255)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--eval-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "eval_log.txt"
    if log_path.exists():
        log_path.unlink()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    _, eval_transform = build_transforms(args.image_size)
    dataset = Isic2019Dataset(Path(args.splits_dir) / f"{args.split}.csv", args.image_dir, transform=eval_transform)
    dataset = build_subset(dataset, args.eval_size, args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_batch,
    )

    model = build_model(len(LABELS), args.pretrained).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    criterion = nn.CrossEntropyLoss()

    config = vars(args)
    config["device"] = str(device)
    write_log(log_path, "实验配置：" + json.dumps(config, ensure_ascii=False, indent=2))

    metrics = {
        "config": config,
        "result": evaluate_attack(model, dataloader, criterion, device, args),
        "method_note": "ISIC 2019 clean baseline 的轻量对抗评测，输出类别与属性分组鲁棒性。",
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    result = metrics["result"]
    anatom = result["adversarial_group_metrics"]["anatom_site_general"]
    write_log(log_path, "clean accuracy: {:.4f}".format(result["clean_accuracy"]))
    write_log(log_path, "adversarial accuracy: {:.4f}".format(result["adversarial_accuracy"]))
    write_log(
        log_path,
        "adversarial worst-class accuracy: {:.4f}".format(result["adversarial_worst_class_accuracy"]),
    )
    write_log(
        log_path,
        "adversarial anatom-site worst-group accuracy: {:.4f}".format(anatom["worst_group_accuracy"]),
    )
    write_log(log_path, f"指标已保存：{metrics_path}")


if __name__ == "__main__":
    main()
