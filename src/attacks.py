import torch


def _linf_project(adv_images, clean_images, epsilon):
    delta = torch.clamp(adv_images - clean_images, min=-epsilon, max=epsilon)
    return torch.clamp(clean_images + delta, 0, 1).detach()


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


def multi_step_fgsm_attack(model, images, labels, criterion, epsilon, alpha, steps):
    model.eval()
    clean_images = images.clone().detach()
    adv_images = clean_images.clone().detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()

        adv_images = adv_images + alpha * adv_images.grad.sign()
        adv_images = _linf_project(adv_images, clean_images, epsilon)

    return adv_images


def untargeted_pgd_attack(model, images, labels, criterion, epsilon, alpha, steps, random_start=True):
    model.eval()
    clean_images = images.clone().detach()
    if random_start:
        noise = torch.empty_like(clean_images).uniform_(-epsilon, epsilon)
        adv_images = torch.clamp(clean_images + noise, 0, 1).detach()
    else:
        adv_images = clean_images.clone().detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()

        adv_images = adv_images + alpha * adv_images.grad.sign()
        adv_images = _linf_project(adv_images, clean_images, epsilon)

    return adv_images


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
        adv_images = _linf_project(adv_images, clean_images, epsilon)

    return adv_images
