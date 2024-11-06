import os
import tempfile

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor, Normalize, Compose

# Model, Loss, Optimizer
model = resnet18(num_classes=10)
model.conv1 = torch.nn.Conv2d(
    1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
)
# model.to("cuda")
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Data
transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
train_data = FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

# Training
for epoch in range(10):
    for images, labels in train_loader:
        images, labels = images.to("cuda"), labels.to("cuda")
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    metrics = {"loss": loss.item(), "epoch": epoch}
    checkpoint_dir = tempfile.mkdtemp()
    checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(metrics)
