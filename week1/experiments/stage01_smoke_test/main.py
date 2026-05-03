import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 下载数据集（CIFAR10）
trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=32,
    shuffle=True
)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*32*32, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

def fgsm_attack(model, images, labels, criterion, epsilon=8/255):
    """生成 untargeted FGSM 对抗样本，用于验证对抗攻击流程。"""
    model.eval()
    adv_images = images.clone().detach().to(device)
    adv_images.requires_grad = True

    outputs = model(adv_images)
    loss = criterion(outputs, labels)

    model.zero_grad()
    loss.backward()

    perturbation = epsilon * adv_images.grad.sign()
    adv_images = adv_images + perturbation
    adv_images = torch.clamp(adv_images, 0, 1).detach()
    return adv_images


def targeted_pgd_attack(model, images, target_labels, criterion, epsilon=8/255, alpha=2/255, steps=5):
    """生成 targeted PGD 对抗样本，作为 FAIR-TAT 目标攻击流程的最小验证。"""
    model.eval()
    clean_images = images.clone().detach().to(device)
    adv_images = clean_images.clone().detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = criterion(outputs, target_labels)

        model.zero_grad()
        loss.backward()

        # targeted attack 需要让模型更接近目标类别，因此沿着降低目标类别损失的方向更新。
        adv_images = adv_images - alpha * adv_images.grad.sign()
        delta = torch.clamp(adv_images - clean_images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(clean_images + delta, 0, 1).detach()

    return adv_images


model = SimpleCNN().to(device)

# 损失函数 + 优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练 1 个 batch，先验证基础训练流程。
for images, labels in trainloader:
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    clean_loss = loss.item()

    # 生成 FGSM 对抗样本，验证对抗攻击流程是否可以接入。
    adv_images = fgsm_attack(model, images, labels, criterion)
    with torch.no_grad():
        adv_outputs = model(adv_images)
        adv_loss = criterion(adv_outputs, labels).item()

    # 用“真实类别 + 1”的方式构造目标类别，验证 targeted PGD 攻击流程。
    target_labels = (labels + 1) % 10
    with torch.no_grad():
        clean_target_loss = criterion(model(images), target_labels).item()

    targeted_adv_images = targeted_pgd_attack(model, images, target_labels, criterion)
    with torch.no_grad():
        targeted_outputs = model(targeted_adv_images)
        targeted_loss = criterion(targeted_outputs, target_labels).item()
        targeted_original_loss = criterion(targeted_outputs, labels).item()

    print("device:", device)
    print("clean loss:", clean_loss)
    print("fgsm adversarial loss:", adv_loss)
    print("clean target loss:", clean_target_loss)
    print("targeted pgd loss to target:", targeted_loss)
    print("targeted pgd loss to original:", targeted_original_loss)
    break  # 只跑一个 batch

print("训练成功！")
print("FGSM 对抗攻击流程验证成功！")
print("targeted PGD 对抗攻击流程验证成功！")

torch.save(model.state_dict(), "model.pth")
print("模型已保存")
