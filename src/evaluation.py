import torch
from contextlib import nullcontext

from src.attacks import (
    fgsm_attack,
    multi_step_fgsm_attack,
    targeted_pgd_attack,
    untargeted_pgd_attack,
)
from src.target_selection import sample_target_labels


def _autocast(enabled):
    if enabled and torch.cuda.is_available():
        return torch.cuda.amp.autocast()
    return nullcontext()


def _as_list(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().tolist()
    return values


def _rate_dict(correct, seen):
    correct = _as_list(correct)
    seen = _as_list(seen)
    return {
        str(cls): (correct[cls] / seen[cls] if seen[cls] > 0 else None)
        for cls in range(len(seen))
    }


def _worst_rate(correct, seen):
    correct = _as_list(correct)
    seen = _as_list(seen)
    rates = [correct[cls] / seen[cls] for cls in range(len(seen)) if seen[cls] > 0]
    return min(rates) if rates else None


def _class_correct_counts(preds, labels, num_classes):
    correct_labels = labels[preds == labels]
    return torch.bincount(correct_labels, minlength=num_classes)


def evaluate_clean(model, dataloader, criterion, device, num_classes, use_amp=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    class_correct = torch.zeros(num_classes, dtype=torch.long, device=device)
    class_seen = torch.zeros(num_classes, dtype=torch.long, device=device)
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with _autocast(use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_seen += labels.size(0)

            class_seen += torch.bincount(labels, minlength=num_classes)
            class_correct += _class_correct_counts(preds, labels, num_classes)
            flat_indices = labels * num_classes + preds
            confusion_matrix += torch.bincount(flat_indices, minlength=num_classes * num_classes).view(
                num_classes,
                num_classes,
            )

    return {
        "loss": total_loss / total_seen,
        "accuracy": total_correct / total_seen,
        "worst_class_accuracy": _worst_rate(class_correct, class_seen),
        "class_accuracy": _rate_dict(class_correct, class_seen),
        "class_seen": {str(cls): int(class_seen[cls].item()) for cls in range(num_classes)},
        "confusion_matrix": confusion_matrix.detach().cpu().tolist(),
    }


def evaluate_robustness(model, dataloader, criterion, device, args, target_prior, use_amp=False):
    model.eval()
    num_classes = args.num_classes
    eval_pgd_steps = getattr(args, "eval_pgd_steps", args.pgd_steps)
    eval_ifgsm_steps = getattr(args, "eval_ifgsm_steps", max(1, args.pgd_steps))
    total_seen = 0
    fgsm_correct = 0
    ifgsm_correct = 0
    pgd_correct = 0
    targeted_correct = 0
    targeted_success = 0
    fgsm_class_correct = torch.zeros(num_classes, dtype=torch.long, device=device)
    ifgsm_class_correct = torch.zeros(num_classes, dtype=torch.long, device=device)
    pgd_class_correct = torch.zeros(num_classes, dtype=torch.long, device=device)
    targeted_class_correct = torch.zeros(num_classes, dtype=torch.long, device=device)
    target_success_class = torch.zeros(num_classes, dtype=torch.long, device=device)
    class_seen = torch.zeros(num_classes, dtype=torch.long, device=device)

    for images, labels in dataloader:
        if total_seen >= args.attack_eval_size:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        target_labels = sample_target_labels(labels, target_prior, num_classes)

        fgsm_images = fgsm_attack(model, images, labels, criterion, args.epsilon, use_amp=use_amp)
        ifgsm_images = multi_step_fgsm_attack(
            model,
            images,
            labels,
            criterion,
            args.epsilon,
            args.alpha,
            eval_ifgsm_steps,
            use_amp=use_amp,
        )
        pgd_images = untargeted_pgd_attack(
            model,
            images,
            labels,
            criterion,
            args.epsilon,
            args.alpha,
            eval_pgd_steps,
            random_start=True,
            use_amp=use_amp,
        )
        targeted_images = targeted_pgd_attack(
            model,
            images,
            target_labels,
            criterion,
            args.epsilon,
            args.alpha,
            args.pgd_steps,
            use_amp=use_amp,
        )

        with torch.no_grad():
            with _autocast(use_amp):
                fgsm_preds = model(fgsm_images).argmax(dim=1)
                ifgsm_preds = model(ifgsm_images).argmax(dim=1)
                pgd_preds = model(pgd_images).argmax(dim=1)
                targeted_preds = model(targeted_images).argmax(dim=1)

        fgsm_correct += (fgsm_preds == labels).sum().item()
        ifgsm_correct += (ifgsm_preds == labels).sum().item()
        pgd_correct += (pgd_preds == labels).sum().item()
        targeted_correct += (targeted_preds == labels).sum().item()
        targeted_success += (targeted_preds == target_labels).sum().item()

        class_seen += torch.bincount(labels, minlength=num_classes)
        fgsm_class_correct += _class_correct_counts(fgsm_preds, labels, num_classes)
        ifgsm_class_correct += _class_correct_counts(ifgsm_preds, labels, num_classes)
        pgd_class_correct += _class_correct_counts(pgd_preds, labels, num_classes)
        targeted_class_correct += _class_correct_counts(targeted_preds, labels, num_classes)
        target_success_class += torch.bincount(labels[targeted_preds == target_labels], minlength=num_classes)

        total_seen += labels.size(0)

    return {
        "attack_eval_size": total_seen,
        "eval_ifgsm_steps": eval_ifgsm_steps,
        "eval_pgd_steps": eval_pgd_steps,
        "fgsm_adversarial_accuracy": fgsm_correct / total_seen,
        "fgsm_worst_class_accuracy": _worst_rate(fgsm_class_correct, class_seen),
        "fgsm_class_accuracy": _rate_dict(fgsm_class_correct, class_seen),
        "ifgsm_adversarial_accuracy": ifgsm_correct / total_seen,
        "ifgsm_worst_class_accuracy": _worst_rate(ifgsm_class_correct, class_seen),
        "ifgsm_class_accuracy": _rate_dict(ifgsm_class_correct, class_seen),
        "untargeted_pgd_adversarial_accuracy": pgd_correct / total_seen,
        "untargeted_pgd_worst_class_accuracy": _worst_rate(pgd_class_correct, class_seen),
        "untargeted_pgd_class_accuracy": _rate_dict(pgd_class_correct, class_seen),
        "targeted_pgd_original_accuracy": targeted_correct / total_seen,
        "targeted_pgd_worst_class_accuracy": _worst_rate(targeted_class_correct, class_seen),
        "targeted_pgd_original_class_accuracy": _rate_dict(targeted_class_correct, class_seen),
        "targeted_pgd_target_success_rate": targeted_success / total_seen,
        "targeted_pgd_target_success_class": _rate_dict(target_success_class, class_seen),
        "class_seen": {str(cls): int(class_seen[cls].item()) for cls in range(num_classes)},
    }
