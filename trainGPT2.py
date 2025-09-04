import os
import json
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import GPT2Tokenizer

# ============================
# Dataset COCO local
# ============================
class COCODataset(Dataset):
    def __init__(self, root_dir, ann_file, transform=None, tokenizer=None, max_len=20):
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(ann_file, "r") as f:
            annotations = json.load(f)

        self.images = []
        self.captions = {}
        for ann in annotations["annotations"]:
            img_id = ann["image_id"]
            caption = ann["caption"]
            self.captions.setdefault(img_id, []).append(caption)

        self.img_id_to_filename = {img["id"]: img["file_name"] for img in annotations["images"]}
        self.img_ids = list(self.captions.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.root_dir, self.img_id_to_filename[img_id])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        caption = self.captions[img_id][0]  # première caption
        tokens = self.tokenizer.encode(caption, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt").squeeze(0)
        return image, tokens

# ============================
# Modèle Encoder + Decoder
# ============================
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
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


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs


class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

# ============================
# Préparation
# ============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size

train_dataset = COCODataset(
    root_dir="COCO/train2017",
    ann_file="COCO/annotations/captions_train2017.json",
    transform=transform,
    tokenizer=tokenizer,
    max_len=20
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

embed_size = 256
hidden_size = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageCaptioningModel(embed_size, hidden_size, vocab_size).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ============================
# Entraînement
# ============================
num_epochs = 3

if __name__ == "__main__":
    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True)

        for i, (imgs, caps) in enumerate(progress_bar):
            imgs = imgs.to(device)
            caps = caps.to(device)

            optimizer.zero_grad()
            outputs = model(imgs, caps[:, :-1])
            loss = criterion(outputs[:, 1:].reshape(-1, vocab_size), caps[:, 1:].reshape(-1))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)
            progress_bar.set_postfix({"loss": avg_loss})

        print(f"Epoch {epoch+1} finished - Average Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), f"checkpoint_epoch{epoch+1}.pth")
        print(f"✅ Modèle sauvegardé: checkpoint_epoch{epoch+1}.pth")
