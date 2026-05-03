import torch


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
