import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_dataset_loaders, get_num_classes
from src.evaluation import evaluate_clean, evaluate_robustness
from src.models import build_model
from src.target_selection import (
    blend_priors,
    class_weights_from_accuracy,
    false_positive_prior,
    robust_deficit_prior,
    uniform_prior,
)
from src.training import train_one_epoch_clean, train_one_epoch_fair_tat_like
from src.utils import set_seed


def write_log(log_path, message):
    print(message)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def select_class_accuracy(clean_metrics, robust_metrics, source):
    if source == "clean":
        return clean_metrics["class_accuracy"]
    if robust_metrics is None:
        return clean_metrics["class_accuracy"]
    source_to_key = {
        "fgsm": "fgsm_class_accuracy",
        "ifgsm": "ifgsm_class_accuracy",
        "untargeted_pgd": "untargeted_pgd_class_accuracy",
        "targeted_pgd": "targeted_pgd_original_class_accuracy",
    }
    return robust_metrics[source_to_key[source]]


def evaluate_prior_source(model, testloader, criterion, device, args, target_prior):
    if args.deficit_source == "clean" or args.prior_eval_size <= 0:
        return None

    original_attack_eval_size = args.attack_eval_size
    args.attack_eval_size = args.prior_eval_size
    robust_metrics = evaluate_robustness(
        model,
        testloader,
        criterion,
        device,
        args,
        target_prior,
        use_amp=args.amp,
    )
    args.attack_eval_size = original_attack_eval_size
    return robust_metrics


def update_target_prior(clean_metrics, deficit_metrics, args, device):
    confusion_matrix = torch.tensor(clean_metrics["confusion_matrix"], device=device)
    false_positive = false_positive_prior(confusion_matrix, smoothing=args.prior_smoothing).to(device)
    class_accuracy = select_class_accuracy(clean_metrics, deficit_metrics, args.deficit_source)
    deficit = robust_deficit_prior(
        class_accuracy,
        args.num_classes,
        device,
        smoothing=args.deficit_smoothing,
    )

    if args.prior_mode == "false_positive":
        return false_positive, deficit, class_accuracy
    if args.prior_mode == "robust_deficit":
        return deficit, deficit, class_accuracy
    if args.prior_mode == "blended":
        return blend_priors(false_positive, deficit, deficit_weight=args.deficit_weight), deficit, class_accuracy
    raise ValueError(f"未知 prior mode：{args.prior_mode}")


def update_class_loss_weights(clean_metrics, deficit_metrics, args, device):
    if args.loss_weight_mode == "none":
        return None, None
    class_accuracy = select_class_accuracy(clean_metrics, deficit_metrics, args.deficit_source)
    weights = class_weights_from_accuracy(
        class_accuracy,
        args.num_classes,
        device,
        strength=args.loss_weight_strength,
    )
    return weights, class_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="FAIR-TAT-like targeted adversarial training")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="week2/experiments/stage01_fair_tat_like_smoke")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--attack-eval-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--model", type=str, default="small_cnn", choices=["small_cnn", "resnet18"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--pgd-steps", type=int, default=3)
    parser.add_argument("--eval-pgd-steps", type=int, default=10)
    parser.add_argument("--eval-ifgsm-steps", type=int, default=5)
    parser.add_argument("--adv-weight", type=float, default=0.5)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--prior-smoothing", type=float, default=1.0)
    parser.add_argument(
        "--prior-mode",
        type=str,
        default="false_positive",
        choices=["false_positive", "robust_deficit", "blended"],
    )
    parser.add_argument(
        "--deficit-source",
        type=str,
        default="clean",
        choices=["clean", "fgsm", "ifgsm", "untargeted_pgd", "targeted_pgd"],
    )
    parser.add_argument("--deficit-smoothing", type=float, default=1.0)
    parser.add_argument("--deficit-weight", type=float, default=0.5)
    parser.add_argument("--prior-eval-size", type=int, default=512)
    parser.add_argument(
        "--loss-weight-mode",
        type=str,
        default="none",
        choices=["none", "robust_deficit"],
    )
    parser.add_argument("--loss-weight-strength", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_classes is None:
        args.num_classes = get_num_classes(args.dataset)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.txt"
    if log_path.exists():
        log_path.unlink()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    use_amp = args.amp and device.type == "cuda"

    trainloader, testloader = build_dataset_loaders(
        args.dataset,
        args.data_dir,
        args.train_size,
        args.test_size,
        args.batch_size,
        args.seed,
        args.num_workers,
        args.pin_memory,
    )

    model = build_model(args.model, num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    target_prior = uniform_prior(args.num_classes, device)
    class_loss_weights = None

    config = vars(args)
    config["device"] = str(device)
    write_log(log_path, "实验配置：" + json.dumps(config, ensure_ascii=False, indent=2))

    history = []
    for epoch in range(1, args.epochs + 1):
        if epoch <= args.warmup_epochs:
            train_metrics = train_one_epoch_clean(
                model,
                trainloader,
                criterion,
                optimizer,
                device,
                use_amp=use_amp,
                scaler=scaler,
            )
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
                class_loss_weights=class_loss_weights,
                use_amp=use_amp,
                scaler=scaler,
            )
            mode = "fair_tat_like"

        clean_metrics = evaluate_clean(model, testloader, criterion, device, args.num_classes, use_amp=use_amp)
        deficit_metrics = evaluate_prior_source(model, testloader, criterion, device, args, target_prior)
        target_prior, deficit_prior, prior_class_accuracy = update_target_prior(
            clean_metrics,
            deficit_metrics,
            args,
            device,
        )
        class_loss_weights, loss_weight_class_accuracy = update_class_loss_weights(
            clean_metrics,
            deficit_metrics,
            args,
            device,
        )

        epoch_metrics = {
            "epoch": epoch,
            "mode": mode,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "clean_test_accuracy": clean_metrics["accuracy"],
            "clean_worst_class_accuracy": clean_metrics["worst_class_accuracy"],
            "target_prior": [round(value, 6) for value in target_prior.detach().cpu().tolist()],
            "deficit_prior": [round(value, 6) for value in deficit_prior.detach().cpu().tolist()],
            "prior_class_accuracy": prior_class_accuracy,
        }
        if class_loss_weights is not None:
            epoch_metrics["class_loss_weights"] = [
                round(value, 6) for value in class_loss_weights.detach().cpu().tolist()
            ]
            epoch_metrics["loss_weight_class_accuracy"] = loss_weight_class_accuracy
        if "clean_loss" in train_metrics:
            epoch_metrics["train_clean_loss"] = train_metrics["clean_loss"]
            epoch_metrics["train_adv_loss"] = train_metrics["adv_loss"]

        history.append(epoch_metrics)
        latest_checkpoint_path = output_dir / "model_latest.pth"
        latest_history_path = output_dir / "history_latest.json"
        torch.save(model.state_dict(), latest_checkpoint_path)
        with latest_history_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        message = (
            f"epoch {epoch}/{args.epochs} | mode: {mode} | "
            f"train loss: {epoch_metrics['train_loss']:.4f} | "
            f"train acc: {epoch_metrics['train_accuracy']:.4f} | "
            f"clean acc: {epoch_metrics['clean_test_accuracy']:.4f} | "
            f"clean worst-class acc: {epoch_metrics['clean_worst_class_accuracy']:.4f} | "
            f"target prior: {epoch_metrics['target_prior']}"
        )
        if class_loss_weights is not None:
            message += f" | class weights: {epoch_metrics['class_loss_weights']}"
        if "train_adv_loss" in epoch_metrics:
            message += f" | adv loss: {epoch_metrics['train_adv_loss']:.4f}"
        write_log(log_path, message)

    clean_metrics = evaluate_clean(model, testloader, criterion, device, args.num_classes, use_amp=use_amp)
    robust_metrics = evaluate_robustness(model, testloader, criterion, device, args, target_prior, use_amp=use_amp)
    metrics = {
        "config": config,
        "history": history,
        "final_target_prior": [round(value, 6) for value in target_prior.detach().cpu().tolist()],
        "final_class_loss_weights": (
            [round(value, 6) for value in class_loss_weights.detach().cpu().tolist()]
            if class_loss_weights is not None
            else None
        ),
        "clean_eval": clean_metrics,
        "robust_eval": robust_metrics,
        "method_note": (
            f"该入口为 FAIR-TAT-like 轻量复现：在 {args.dataset} 上使用 class false positive prior "
            "与 robust deficit prior 采样 targeted PGD 的目标类别。"
        ),
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
    write_log(log_path, "multi-step FGSM robust accuracy: {:.4f}".format(robust_metrics["ifgsm_adversarial_accuracy"]))
    write_log(
        log_path,
        "multi-step FGSM worst-class robust accuracy: {:.4f}".format(robust_metrics["ifgsm_worst_class_accuracy"]),
    )
    write_log(
        log_path,
        "untargeted PGD robust accuracy: {:.4f}".format(robust_metrics["untargeted_pgd_adversarial_accuracy"]),
    )
    write_log(
        log_path,
        "untargeted PGD worst-class robust accuracy: {:.4f}".format(
            robust_metrics["untargeted_pgd_worst_class_accuracy"]
        ),
    )
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
