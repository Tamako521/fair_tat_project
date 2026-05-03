import random

import torch


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def accuracy_from_logits(outputs, labels):
    preds = outputs.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct, labels.size(0)
