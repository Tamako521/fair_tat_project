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


def sample_target_labels(labels, prior, num_classes):
    targets = []
    for label in labels.detach().cpu().tolist():
        probs = prior.detach().clone()
        probs[label] = 0.0
        if probs.sum() <= 0:
            probs = torch.ones(num_classes, device=prior.device)
            probs[label] = 0.0
        probs = probs / probs.sum()
        sampled = torch.multinomial(probs, num_samples=1).item()
        targets.append(sampled)
    return torch.tensor(targets, dtype=labels.dtype, device=labels.device)
