import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from train_cifar10_baseline import SmallCNN, build_subset, fgsm_attack, targeted_pgd_attack


def evaluate_adversarial_classwise(model, testloader, criterion, device, args):
    model.eval()
    num_classes = 10
    class_seen = [0 for _ in range(num_classes)]
    fgsm_correct = [0 for _ in range(num_classes)]
    targeted_original_correct = [0 for _ in range(num_classes)]
    targeted_success = [0 for _ in range(num_classes)]
    total_seen = 0

    for images, labels in testloader:
        if total_seen >= args.attack_eval_size:
            break

        images = images.to(device)
        labels = labels.to(device)
        target_labels = (labels + 1) % num_classes

        fgsm_images = fgsm_attack(model, images, labels, criterion, args.epsilon)
        targeted_images = targeted_pgd_attack(
            model,
            images,
            target_labels,
            criterion,
            args.epsilon,
            args.alpha,
            args.pgd_steps,
        )

        with torch.no_grad():
            fgsm_preds = model(fgsm_images).argmax(dim=1)
            targeted_preds = model(targeted_images).argmax(dim=1)

        for cls in range(num_classes):
            mask = labels == cls
            count = mask.sum().item()
            class_seen[cls] += count
            fgsm_correct[cls] += ((fgsm_preds == labels) & mask).sum().item()
            targeted_original_correct[cls] += ((targeted_preds == labels) & mask).sum().item()
            targeted_success[cls] += ((targeted_preds == target_labels) & mask).sum().item()

        total_seen += labels.size(0)

    def to_rate_dict(values):
        return {
            str(cls): (values[cls] / class_seen[cls] if class_seen[cls] > 0 else None)
            for cls in range(num_classes)
        }

    return {
        "class_seen": {str(cls): class_seen[cls] for cls in range(num_classes)},
        "fgsm_class_accuracy": to_rate_dict(fgsm_correct),
        "targeted_pgd_original_class_accuracy": to_rate_dict(targeted_original_correct),
        "targeted_pgd_target_success_rate": to_rate_dict(targeted_success),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="分析 CIFAR-10 对抗评测的类别级表现")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="docs")
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--attack-eval-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--pgd-steps", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])

    testset = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform,
    )
    test_subset = build_subset(testset, args.test_size, args.seed + 1)
    testloader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SmallCNN().to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    criterion = nn.CrossEntropyLoss()

    result = {
        "experiment_name": args.experiment_name,
        "model_path": args.model_path,
        "config": {
            "test_size": args.test_size,
            "attack_eval_size": args.attack_eval_size,
            "epsilon": args.epsilon,
            "alpha": args.alpha,
            "pgd_steps": args.pgd_steps,
            "device": str(device),
        },
        "adversarial_classwise": evaluate_adversarial_classwise(
            model,
            testloader,
            criterion,
            device,
            args,
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.experiment_name}_adversarial_classwise.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"类别级对抗评测结果已保存：{output_path}")


if __name__ == "__main__":
    main()

