from src.attacks import targeted_pgd_attack
from src.target_selection import sample_target_labels
from src.utils import accuracy_from_logits


def train_one_epoch_clean(model, trainloader, criterion, optimizer, device):
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


def train_one_epoch_fair_tat_like(model, trainloader, criterion, optimizer, device, args, target_prior):
    model.train()
    total_loss = 0.0
    total_clean_loss = 0.0
    total_adv_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, labels in trainloader:
        images = images.to(device)
        labels = labels.to(device)
        target_labels = sample_target_labels(labels, target_prior, args.num_classes)

        targeted_images = targeted_pgd_attack(
            model,
            images,
            target_labels,
            criterion,
            args.epsilon,
            args.alpha,
            args.pgd_steps,
        )

        model.train()
        clean_outputs = model(images)
        adv_outputs = model(targeted_images)
        clean_loss = criterion(clean_outputs, labels)
        adv_loss = criterion(adv_outputs, labels)
        loss = (1 - args.adv_weight) * clean_loss + args.adv_weight * adv_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct, seen = accuracy_from_logits(clean_outputs, labels)
        total_loss += loss.item() * seen
        total_clean_loss += clean_loss.item() * seen
        total_adv_loss += adv_loss.item() * seen
        total_correct += correct
        total_seen += seen

    return {
        "loss": total_loss / total_seen,
        "clean_loss": total_clean_loss / total_seen,
        "adv_loss": total_adv_loss / total_seen,
        "accuracy": total_correct / total_seen,
    }
