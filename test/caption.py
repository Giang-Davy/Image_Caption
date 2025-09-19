#!/usr/bin/env python3

import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from transformers import GPT2Tokenizer

# ============================
# Define the model
# ============================
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
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
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
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
# Settings
# ============================
embed_size = 256
hidden_size = 512
vocab_size = 50257  # GPT-2 vocab size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# Load the model and the tokenizer
# ============================
model = ImageCaptioningModel(embed_size, hidden_size, vocab_size).to(device)
checkpoint_path = "checkpoint_epoch3.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# ============================
# Preparing the image
# ============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
image_path = "surf.webp"
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# ============================
# Generate the caption
# ============================
input_ids = torch.tensor([tokenizer.bos_token_id]).unsqueeze(0).to(device)

max_len = 20
features = model.encoder(image)
caption_tokens = input_ids

for _ in range(max_len):
    outputs = model.decoder(features, caption_tokens)
    next_token_logits = outputs[:, -1, :]
    next_token_id = next_token_logits.argmax(1).unsqueeze(0)
    caption_tokens = torch.cat((caption_tokens, next_token_id), dim=1)

caption_text = tokenizer.decode(caption_tokens.squeeze(), skip_special_tokens=True)
print("Generated caption:", caption_text)
