import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_cifar10_loaders
from src.evaluation import evaluate_clean, evaluate_robustness
from src.models import SmallCNN
from src.target_selection import false_positive_prior, uniform_prior
from src.training import train_one_epoch_clean, train_one_epoch_fair_tat_like
from src.utils import set_seed


def write_log(log_path, message):
    print(message)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="FAIR-TAT-like CIFAR-10 targeted adversarial training")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="week2/experiments/stage01_fair_tat_like_smoke")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--attack-eval-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--pgd-steps", type=int, default=3)
    parser.add_argument("--adv-weight", type=float, default=0.5)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--prior-smoothing", type=float, default=1.0)
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
    trainloader, testloader = build_cifar10_loaders(
        args.data_dir,
        args.train_size,
        args.test_size,
        args.batch_size,
        args.seed,
    )

    model = SmallCNN(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    target_prior = uniform_prior(args.num_classes, device)

    config = vars(args)
    config["device"] = str(device)
    write_log(log_path, "实验配置：" + json.dumps(config, ensure_ascii=False, indent=2))

    history = []
    for epoch in range(1, args.epochs + 1):
        if epoch <= args.warmup_epochs:
            train_metrics = train_one_epoch_clean(model, trainloader, criterion, optimizer, device)
            mode = "clean_warmup"
        else:
            train_metrics = train_one_epoch_fair_tat_like(
                model,
                trainloader,
                criterion,
                optimizer,
                device,
                args,
                target_prior,
            )
            mode = "fair_tat_like"

        clean_metrics = evaluate_clean(model, testloader, criterion, device, args.num_classes)
        confusion_matrix = torch.tensor(clean_metrics["confusion_matrix"], device=device)
        target_prior = false_positive_prior(confusion_matrix, smoothing=args.prior_smoothing).to(device)

        epoch_metrics = {
            "epoch": epoch,
            "mode": mode,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "clean_test_accuracy": clean_metrics["accuracy"],
            "clean_worst_class_accuracy": clean_metrics["worst_class_accuracy"],
            "target_prior": [round(value, 6) for value in target_prior.detach().cpu().tolist()],
        }
        if "clean_loss" in train_metrics:
            epoch_metrics["train_clean_loss"] = train_metrics["clean_loss"]
            epoch_metrics["train_adv_loss"] = train_metrics["adv_loss"]

        history.append(epoch_metrics)
        message = (
            f"epoch {epoch}/{args.epochs} | mode: {mode} | "
            f"train loss: {epoch_metrics['train_loss']:.4f} | "
            f"train acc: {epoch_metrics['train_accuracy']:.4f} | "
            f"clean acc: {epoch_metrics['clean_test_accuracy']:.4f} | "
            f"clean worst-class acc: {epoch_metrics['clean_worst_class_accuracy']:.4f} | "
            f"target prior: {epoch_metrics['target_prior']}"
        )
        if "train_adv_loss" in epoch_metrics:
            message += f" | adv loss: {epoch_metrics['train_adv_loss']:.4f}"
        write_log(log_path, message)

    clean_metrics = evaluate_clean(model, testloader, criterion, device, args.num_classes)
    robust_metrics = evaluate_robustness(model, testloader, criterion, device, args, target_prior)
    metrics = {
        "config": config,
        "history": history,
        "final_target_prior": [round(value, 6) for value in target_prior.detach().cpu().tolist()],
        "clean_eval": clean_metrics,
        "robust_eval": robust_metrics,
        "method_note": "该入口为 FAIR-TAT-like 轻量复现：使用 class false positive prior 采样 targeted PGD 的目标类别。",
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    model_path = output_dir / "model.pth"
    torch.save(model.state_dict(), model_path)

    write_log(log_path, "最终 clean accuracy: {:.4f}".format(clean_metrics["accuracy"]))
    write_log(log_path, "最终 clean worst-class accuracy: {:.4f}".format(clean_metrics["worst_class_accuracy"]))
    write_log(log_path, "FGSM robust accuracy: {:.4f}".format(robust_metrics["fgsm_adversarial_accuracy"]))
    write_log(log_path, "FGSM worst-class robust accuracy: {:.4f}".format(robust_metrics["fgsm_worst_class_accuracy"]))
    write_log(
        log_path,
        "targeted PGD original accuracy: {:.4f}".format(robust_metrics["targeted_pgd_original_accuracy"]),
    )
    write_log(
        log_path,
        "targeted PGD worst-class robust accuracy: {:.4f}".format(
            robust_metrics["targeted_pgd_worst_class_accuracy"]
        ),
    )
    write_log(
        log_path,
        "targeted PGD target success rate: {:.4f}".format(robust_metrics["targeted_pgd_target_success_rate"]),
    )
    write_log(log_path, f"指标已保存：{metrics_path}")
    write_log(log_path, f"模型已保存：{model_path}")


if __name__ == "__main__":
    main()
