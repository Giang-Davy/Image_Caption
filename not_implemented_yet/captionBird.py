#!/usr/bin/env python3

import torch
from PIL import Image
from torchvision import transforms, models
from transformers import GPT2Tokenizer
import torch.nn as nn
from bird_names import bird_names

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (224, 224)
EMBED_SIZE = 256
HIDDEN_SIZE = 512
MAX_LEN = 20

# ====== Modèle Show-and-Tell ======
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
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
        super().__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

# ====== Tokenizer et transform ======
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])

# ====== Charger les checkpoints ======
model_caption = ImageCaptioningModel(EMBED_SIZE, HIDDEN_SIZE, vocab_size).to(device)
model_caption.load_state_dict(torch.load("checkpoint_epoch3.pth", map_location=device))
model_caption.eval()

# Classifieur oiseaux (embedding 256)
class BirdClassifier(nn.Module):
    def __init__(self, embed_size=EMBED_SIZE, num_classes=200):
        super().__init__()
        self.linear = nn.Linear(embed_size, num_classes)  # directement 256 -> 200

    def forward(self, x):
        return self.linear(x)

classifier_bird = BirdClassifier().to(device)
classifier_bird.load_state_dict(torch.load("bird_classifier_256.pth", map_location=device))
classifier_bird.eval()

# ====== Génération caption avec remplacement ======
def generate_caption_with_species(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Encoder
    features = model_caption.encoder(image)

    # Caption général
    input_ids = torch.tensor([tokenizer.bos_token_id]).unsqueeze(0).to(device)
    caption_tokens = input_ids
    for _ in range(MAX_LEN):
        outputs = model_caption.decoder(features, caption_tokens)
        next_token_id = outputs[:, -1, :].argmax(1).unsqueeze(0)
        caption_tokens = torch.cat((caption_tokens, next_token_id), dim=1)
        if next_token_id.item() == tokenizer.eos_token_id:
            break
    general_caption = tokenizer.decode(caption_tokens.squeeze(), skip_special_tokens=True)

    # Prédire l'espèce
    bird_logits = classifier_bird(features)
    bird_class = torch.argmax(bird_logits, dim=1).item() + 1
    bird_name = bird_names[bird_class]

    # Remplacer "oiseau" par le nom de l'espèce
    final_caption = general_caption.replace("bird", bird_name)

    return final_caption

# ====== Exemple ======
if __name__ == "__main__":
    img_path = "bird1.jpg"
    caption = generate_caption_with_species(img_path)
    print(f"Generated caption: {caption}")
