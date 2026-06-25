import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")

BATCH_SIZE = 128
EPOCHS = 7
LR = 1e-3
WEIGHT_DECAY = 1e-4
SAVE_DIR = Path("output")
SAVE_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.use_skip = stride != 1 or in_c != out_c
        if self.use_skip:
            self.skip = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c),
            )
        self.dw = nn.Conv2d(in_c, in_c, 3, stride, padding=1, groups=in_c, bias=False)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.dw(x)))
        x = self.bn2(self.pw(x))
        if self.use_skip:
            identity = self.skip(identity)
        x = self.relu(x + identity)
        return x


class DepthwiseMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.stages = nn.Sequential(
            DepthwiseSeparableConv(48, 96, stride=2),
            DepthwiseSeparableConv(96, 96, stride=1),
            DepthwiseSeparableConv(96, 192, stride=2),
            DepthwiseSeparableConv(192, 192, stride=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stages(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=False, transform=transform,
    )
    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=False, transform=transform,
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False,
    )
    return train_loader, test_loader


def plot_curves(step_losses, epoch_edges, train_accs, test_accs, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    steps = range(len(step_losses))
    ax1.plot(steps, step_losses, alpha=0.6, linewidth=0.8, label="Step Loss")
    for e in epoch_edges:
        ax1.axvline(x=e, color='r', linestyle='--', alpha=0.4, linewidth=0.7)
    ax1.set_xlabel("Step"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(train_accs, marker='o', label="Train Acc")
    ax2.plot(test_accs, marker='s', label="Val Acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)"); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    model = DepthwiseMNIST().to(DEVICE)
    total = count_params(model)
    print(f"Total params: {total:,}")
    assert 80_000 <= total <= 120_000

    train_loader, test_loader = get_dataloaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    step_losses = []
    epoch_edges = []
    train_accs, test_accs = [], []
    best_acc = 0.0
    steps_so_far = 0

    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        correct = 0
        total_samples = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            step_losses.append(loss.item())
            steps_so_far += 1

            _, predicted = outputs.max(1)
            total_samples += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_edges.append(steps_so_far)
        train_acc = 100.0 * correct / total_samples

        model.eval()
        correct = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total_samples += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = 100.0 * correct / total_samples
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), SAVE_DIR / "best_model.pth")

        print(f"Epoch {epoch}/{EPOCHS} | Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {test_acc:.2f}% | Best: {best_acc:.2f}%")

        scheduler.step()

    t_total = time.time() - t_start
    print(f"\nTotal training time: {t_total:.1f}s")

    plot_curves(step_losses, epoch_edges, train_accs, test_accs, SAVE_DIR / "loss_curves.png")
    print(f"Saved: {SAVE_DIR / 'best_model.pth'}")
    print(f"Saved: {SAVE_DIR / 'loss_curves.png'}")


if __name__ == "__main__":
    main()
