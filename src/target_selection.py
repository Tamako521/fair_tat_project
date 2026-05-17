import torch


def uniform_prior(num_classes, device):
    return torch.full((num_classes,), 1.0 / num_classes, device=device)


def false_positive_prior(confusion_matrix, smoothing=1.0):
    matrix = confusion_matrix.float().clone()
    matrix.fill_diagonal_(0)
    false_positive_counts = matrix.sum(dim=0) + smoothing
    total = false_positive_counts.sum()
    if total <= 0:
        return torch.full_like(false_positive_counts, 1.0 / false_positive_counts.numel())
    return false_positive_counts / total


def accuracy_dict_to_tensor(class_accuracy, num_classes, device):
    values = []
    for cls in range(num_classes):
        value = class_accuracy.get(str(cls), None)
        if value is None:
            value = 0.0
        values.append(float(value))
    return torch.tensor(values, dtype=torch.float32, device=device)


def robust_deficit_prior(class_accuracy, num_classes, device, smoothing=1.0):
    accuracies = accuracy_dict_to_tensor(class_accuracy, num_classes, device)
    deficits = torch.clamp(1.0 - accuracies, min=0.0) + smoothing
    total = deficits.sum()
    if total <= 0:
        return uniform_prior(num_classes, device)
    return deficits / total


def blend_priors(base_prior, deficit_prior, deficit_weight=0.5):
    weight = max(0.0, min(1.0, float(deficit_weight)))
    prior = (1.0 - weight) * base_prior + weight * deficit_prior
    total = prior.sum()
    if total <= 0:
        return torch.full_like(prior, 1.0 / prior.numel())
    return prior / total


def class_weights_from_accuracy(class_accuracy, num_classes, device, strength=1.0):
    accuracies = accuracy_dict_to_tensor(class_accuracy, num_classes, device)
    deficits = torch.clamp(1.0 - accuracies, min=0.0)
    weights = 1.0 + float(strength) * deficits
    return weights / weights.mean().clamp_min(1e-8)


def sample_target_labels(labels, prior, num_classes):
    batch_size = labels.size(0)
    probs = prior.detach().to(labels.device).unsqueeze(0).expand(batch_size, -1).clone()
    probs.scatter_(1, labels.view(-1, 1), 0.0)
    row_sums = probs.sum(dim=1, keepdim=True)
    fallback = torch.ones_like(probs)
    fallback.scatter_(1, labels.view(-1, 1), 0.0)
    probs = torch.where(row_sums > 0, probs, fallback)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return torch.multinomial(probs, num_samples=1).squeeze(1).to(dtype=labels.dtype)
