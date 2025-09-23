#!/usr/bin/env python3
import os
import json
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import GPT2Tokenizer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# === Dataset COCO Validation ===
class COCODataset(Dataset):
    def __init__(self, root_dir, ann_file, transform=None, tokenizer=None, max_len=20):
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(ann_file, "r") as f:
            annotations = json.load(f)

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

        return img_id, image

# === Modèle ===
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

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
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
        super().__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def generate_caption(self, image, tokenizer, max_len=20):
        self.eval()
        device = next(self.parameters()).device
        image = image.unsqueeze(0).to(device)
        feature = self.encoder(image)
        caption_ids = []
        states = None
        inputs = feature.unsqueeze(1)
        for _ in range(max_len):
            hiddens, states = self.decoder.lstm(inputs, states)
            output = self.decoder.linear(hiddens.squeeze(1))
            predicted = output.argmax(1)
            word_id = predicted.item()
            caption_ids.append(word_id)
            inputs = self.decoder.embed(predicted).unsqueeze(1)
            if word_id == tokenizer.eos_token_id:
                break
        return tokenizer.decode(caption_ids, skip_special_tokens=True)

# === Évaluation BLEU ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_size = 256
    hidden_size = 512
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    val_dataset = COCODataset(
        root_dir="COCO/val2017",
        ann_file="COCO/annotations/captions_val2017.json",
        transform=transform,
        tokenizer=tokenizer,
        max_len=20
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Charger modèle
    model = ImageCaptioningModel(embed_size, hidden_size, vocab_size).to(device)
    model.load_state_dict(torch.load("checkpoint_epoch3.pth", map_location=device))
    model.eval()

    # Charger les références
    with open("COCO/annotations/captions_val2017.json", "r") as f:
        annotations = json.load(f)
    captions_dict = {}
    for ann in annotations["annotations"]:
        captions_dict.setdefault(ann["image_id"], []).append(ann["caption"])

    references = []
    hypotheses = []
    smooth = SmoothingFunction().method1

    for img_id, image in tqdm(val_loader, desc="BLEU Evaluation"):
        img_id = img_id.item()
        candidate = model.generate_caption(image[0], tokenizer, max_len=20)
        hypotheses.append(candidate.split())
        refs = [ref.split() for ref in captions_dict[img_id]]
        references.append(refs)

    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    print("\n=== BLEU Scores ===")
    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-2: {bleu2:.4f}")
    print(f"BLEU-3: {bleu3:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")
