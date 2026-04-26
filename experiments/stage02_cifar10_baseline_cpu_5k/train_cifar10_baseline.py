import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def build_subset(dataset, size, seed):
    if size <= 0 or size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def fgsm_attack(model, images, labels, criterion, epsilon):
    model.eval()
    adv_images = images.clone().detach()
    adv_images.requires_grad = True

    outputs = model(adv_images)
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()

    adv_images = adv_images + epsilon * adv_images.grad.sign()
    return torch.clamp(adv_images, 0, 1).detach()


def targeted_pgd_attack(model, images, target_labels, criterion, epsilon, alpha, steps):
    model.eval()
    clean_images = images.clone().detach()
    adv_images = clean_images.clone().detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = criterion(outputs, target_labels)
        model.zero_grad()
        loss.backward()

        adv_images = adv_images - alpha * adv_images.grad.sign()
        delta = torch.clamp(adv_images - clean_images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(clean_images + delta, 0, 1).detach()

    return adv_images


def accuracy_from_logits(outputs, labels):
    preds = outputs.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct, labels.size(0)


def train_one_epoch(model, trainloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, labels in trainloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct, seen = accuracy_from_logits(outputs, labels)
        total_loss += loss.item() * seen
        total_correct += correct
        total_seen += seen

    return {
        "loss": total_loss / total_seen,
        "accuracy": total_correct / total_seen,
    }


def evaluate_clean(model, testloader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    class_correct = [0 for _ in range(num_classes)]
    class_seen = [0 for _ in range(num_classes)]

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_seen += labels.size(0)

            for cls in range(num_classes):
                mask = labels == cls
                class_seen[cls] += mask.sum().item()
                class_correct[cls] += ((preds == labels) & mask).sum().item()

    class_accuracy = {
        str(cls): (class_correct[cls] / class_seen[cls] if class_seen[cls] > 0 else None)
        for cls in range(num_classes)
    }
    valid_class_acc = [value for value in class_accuracy.values() if value is not None]

    return {
        "loss": total_loss / total_seen,
        "accuracy": total_correct / total_seen,
        "worst_class_accuracy": min(valid_class_acc),
        "class_accuracy": class_accuracy,
    }


def evaluate_attacks(model, testloader, criterion, device, args):
    model.eval()
    fgsm_correct = 0
    targeted_pgd_correct = 0
    targeted_success = 0
    total_seen = 0

    for images, labels in testloader:
        if total_seen >= args.attack_eval_size:
            break

        images = images.to(device)
        labels = labels.to(device)
        target_labels = (labels + 1) % 10

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

        fgsm_correct += (fgsm_preds == labels).sum().item()
        targeted_pgd_correct += (targeted_preds == labels).sum().item()
        targeted_success += (targeted_preds == target_labels).sum().item()
        total_seen += labels.size(0)

    return {
        "attack_eval_size": total_seen,
        "fgsm_adversarial_accuracy": fgsm_correct / total_seen,
        "targeted_pgd_original_accuracy": targeted_pgd_correct / total_seen,
        "targeted_pgd_target_success_rate": targeted_success / total_seen,
    }


def write_log(log_path, message):
    print(message)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 小型 baseline 训练与对抗评测")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="experiments/stage02_cifar10_baseline_cpu_5k")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--attack-eval-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--pgd-steps", type=int, default=5)
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
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    testset = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_subset = build_subset(trainset, args.train_size, args.seed)
    test_subset = build_subset(testset, args.test_size, args.seed + 1)

    trainloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    config = vars(args)
    config["device"] = str(device)
    write_log(log_path, "实验配置：" + json.dumps(config, ensure_ascii=False, indent=2))

    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, trainloader, criterion, optimizer, device)
        clean_metrics = evaluate_clean(model, testloader, criterion, device, num_classes=10)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "clean_test_loss": clean_metrics["loss"],
            "clean_test_accuracy": clean_metrics["accuracy"],
            "worst_class_accuracy": clean_metrics["worst_class_accuracy"],
        }
        history.append(epoch_metrics)
        write_log(
            log_path,
            (
                f"epoch {epoch}/{args.epochs} | "
                f"train loss: {epoch_metrics['train_loss']:.4f} | "
                f"train acc: {epoch_metrics['train_accuracy']:.4f} | "
                f"clean test acc: {epoch_metrics['clean_test_accuracy']:.4f} | "
                f"worst-class acc: {epoch_metrics['worst_class_accuracy']:.4f}"
            ),
        )

    clean_metrics = evaluate_clean(model, testloader, criterion, device, num_classes=10)
    attack_metrics = evaluate_attacks(model, testloader, criterion, device, args)
    metrics = {
        "config": config,
        "history": history,
        "clean_eval": clean_metrics,
        "attack_eval": attack_metrics,
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    model_path = output_dir / "model.pth"
    torch.save(model.state_dict(), model_path)

    write_log(log_path, "最终 clean accuracy: {:.4f}".format(clean_metrics["accuracy"]))
    write_log(log_path, "最终 worst-class accuracy: {:.4f}".format(clean_metrics["worst_class_accuracy"]))
    write_log(log_path, "FGSM adversarial accuracy: {:.4f}".format(attack_metrics["fgsm_adversarial_accuracy"]))
    write_log(
        log_path,
        "targeted PGD original accuracy: {:.4f}".format(attack_metrics["targeted_pgd_original_accuracy"]),
    )
    write_log(
        log_path,
        "targeted PGD target success rate: {:.4f}".format(attack_metrics["targeted_pgd_target_success_rate"]),
    )
    write_log(log_path, f"指标已保存：{metrics_path}")
    write_log(log_path, f"模型已保存：{model_path}")


if __name__ == "__main__":
    main()

