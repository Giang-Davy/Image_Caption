#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100
NUM_CLASSES = 200
EMBED_SIZE = 256

# ====== EncoderCNN ======
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
            features = features.reshape(features.size(0), -1)
            features = self.bn(self.linear(features))
        return features

# ====== Classifer ======
class BirdClassifier(nn.Module):
    def __init__(self, embed_size=EMBED_SIZE, num_classes=NUM_CLASSES):
        super().__init__()
        self.linear = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        return self.linear(x)

# ====== Transformations ======
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])

# ====== Dataset ======
train_dataset = datasets.ImageFolder(root="CUB_200_2011/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = datasets.ImageFolder(root="CUB_200_2011/test", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ====== Initialization ======
encoder = EncoderCNN(EMBED_SIZE).to(device)
classifier = BirdClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=1e-4)

# ====== Train ======
for epoch in range(EPOCHS):
    classifier.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        features = encoder(images)
        outputs = classifier(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {train_loss:.4f} - Acc: {train_acc:.4f}")

# ====== Save ======
torch.save(classifier.state_dict(), "bird_classifier_256.pth")
print("Checkpoint bird_classifier_256.pth sauvegardé ✅")
