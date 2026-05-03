import torch

from src.attacks import fgsm_attack, targeted_pgd_attack
from src.target_selection import sample_target_labels


def _rate_dict(correct, seen):
    return {
        str(cls): (correct[cls] / seen[cls] if seen[cls] > 0 else None)
        for cls in range(len(seen))
    }


def _worst_rate(correct, seen):
    rates = [correct[cls] / seen[cls] for cls in range(len(seen)) if seen[cls] > 0]
    return min(rates) if rates else None


def evaluate_clean(model, dataloader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    class_correct = [0 for _ in range(num_classes)]
    class_seen = [0 for _ in range(num_classes)]
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_seen += labels.size(0)

            for true_label, pred_label in zip(labels.cpu(), preds.cpu()):
                cls = int(true_label.item())
                pred = int(pred_label.item())
                class_seen[cls] += 1
                class_correct[cls] += int(cls == pred)
                confusion_matrix[cls, pred] += 1

    return {
        "loss": total_loss / total_seen,
        "accuracy": total_correct / total_seen,
        "worst_class_accuracy": _worst_rate(class_correct, class_seen),
        "class_accuracy": _rate_dict(class_correct, class_seen),
        "class_seen": {str(cls): class_seen[cls] for cls in range(num_classes)},
        "confusion_matrix": confusion_matrix.tolist(),
    }


def evaluate_robustness(model, dataloader, criterion, device, args, target_prior):
    model.eval()
    num_classes = args.num_classes
    total_seen = 0
    fgsm_correct = 0
    targeted_correct = 0
    targeted_success = 0
    fgsm_class_correct = [0 for _ in range(num_classes)]
    targeted_class_correct = [0 for _ in range(num_classes)]
    target_success_class = [0 for _ in range(num_classes)]
    class_seen = [0 for _ in range(num_classes)]

    for images, labels in dataloader:
        if total_seen >= args.attack_eval_size:
            break

        images = images.to(device)
        labels = labels.to(device)
        target_labels = sample_target_labels(labels, target_prior, num_classes)

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
        targeted_correct += (targeted_preds == labels).sum().item()
        targeted_success += (targeted_preds == target_labels).sum().item()

        for cls in range(num_classes):
            mask = labels == cls
            count = mask.sum().item()
            class_seen[cls] += count
            fgsm_class_correct[cls] += ((fgsm_preds == labels) & mask).sum().item()
            targeted_class_correct[cls] += ((targeted_preds == labels) & mask).sum().item()
            target_success_class[cls] += ((targeted_preds == target_labels) & mask).sum().item()

        total_seen += labels.size(0)

    return {
        "attack_eval_size": total_seen,
        "fgsm_adversarial_accuracy": fgsm_correct / total_seen,
        "fgsm_worst_class_accuracy": _worst_rate(fgsm_class_correct, class_seen),
        "fgsm_class_accuracy": _rate_dict(fgsm_class_correct, class_seen),
        "targeted_pgd_original_accuracy": targeted_correct / total_seen,
        "targeted_pgd_worst_class_accuracy": _worst_rate(targeted_class_correct, class_seen),
        "targeted_pgd_original_class_accuracy": _rate_dict(targeted_class_correct, class_seen),
        "targeted_pgd_target_success_rate": targeted_success / total_seen,
        "targeted_pgd_target_success_class": _rate_dict(target_success_class, class_seen),
        "class_seen": {str(cls): class_seen[cls] for cls in range(num_classes)},
    }
