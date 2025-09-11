#!/usr/bin/env python3
from flask import Flask, request, render_template_string
from pymongo import MongoClient
import gridfs
import base64
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
from transformers import GPT2Tokenizer
import ast

with open("food_list.txt", "r", encoding="utf-8") as f:
    contenu = f.read()
list_food = ast.literal_eval(contenu)

animal_list = ["dog", "dogs", "cat", "cats", "horse", "horses", "cow", "cows", "sheep", "goat", "goats",
"pig", "pigs", "chicken", "chickens", "duck", "ducks", "bird", "birds", "rabbit", "rabbits",
"mouse", "mice", "rat", "rats", "elephant", "elephants", "lion", "lions", "tiger", "tigers",
"bear", "bears", "zebra", "zebras", "giraffe", "giraffes", "monkey", "monkeys", "ape", "apes",
"gorilla", "gorillas", "kangaroo", "kangaroos", "wolf", "wolves", "fox", "foxes", "deer",
"camel", "camels", "donkey", "donkeys", "buffalo", "buffaloes", "leopard", "leopards",
"cheetah", "cheetahs", "crocodile", "crocodiles", "alligator", "alligators", "snake", "snakes",
"lizard", "lizards", "turtle", "turtles", "frog", "frogs", "fish", "whale", "whales",
"dolphin", "dolphins", "shark", "sharks", "seal", "seals", "penguin", "penguins", "owl", "owls",
"eagle", "eagles", "hawk", "hawks", "parrot", "parrots", "goose", "geese", "turkey", "turkeys",
"swan", "swans"]

# === Modèle Caption ===
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

# === Préparation modèle ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_size = 256
hidden_size = 512
vocab_size = 50257
model = ImageCaptioningModel(embed_size, hidden_size, vocab_size).to(device)
model.load_state_dict(torch.load("checkpoint_epoch3.pth", map_location=device))
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Flask + MongoDB ===
app = Flask(__name__)
client = MongoClient("mongodb://localhost:27017/")
db = client["Data"]
fs = gridfs.GridFS(db)
captions_collection = db["captions"]
animals_collection = db["animals"]
food_collection = db["foods"]

# === Routes ===
@app.route("/", methods=["GET"])
def index():
    return render_template_string("""
        <h2>Upload Image</h2>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="submit" value="Upload">
        </form>
        <a href="/images">Voir toutes les images</a>
    """)

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files or request.files["file"].filename == "":
        return "No file uploaded", 400

    file = request.files["file"]

    # Stockage dans GridFS
    file_id = fs.put(file, filename=file.filename)

    # Récupération de l'image depuis GridFS pour le modèle
    grid_out = fs.get(file_id)
    image = Image.open(grid_out).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Génération de la caption
    input_ids = torch.tensor([tokenizer.bos_token_id]).unsqueeze(0).to(device)
    max_len = 20
    features = model.encoder(image_tensor)
    caption_tokens = input_ids
    for _ in range(max_len):
        outputs = model.decoder(features, caption_tokens)
        next_token_logits = outputs[:, -1, :]
        next_token_id = next_token_logits.argmax(1).unsqueeze(0)
        caption_tokens = torch.cat((caption_tokens, next_token_id), dim=1)
    caption_text = tokenizer.decode(caption_tokens.squeeze(), skip_special_tokens=True)

    # Stocker la caption dans MongoDB
    captions_collection.insert_one({"file_id": file_id, "caption": caption_text})
    caption_text = caption_text.lower()  # mettre en minuscule pour éviter les problèmes de casse
    caption_words = caption_text.split()
    is_animal = any(mot in caption_words for mot in animal_list) # vérifier si un mot de la liste des animaux se trouve dans la caption et non une sous-chaîne 
    is_food = False

    for mot_food in list_food:
        if mot_food in caption_text:
            is_food = True
            break
    if is_food:
        food_collection.insert_one({
            "file_id": file_id,
            "caption": caption_text,
            "is_food": True
        })


    # Logique de retour corrigée
    if is_animal and not is_food:
        return f"ANIMAL File uploaded and caption generated:<br><b>{caption_text}</b><br><a href='/'>Retour</a>"
    elif is_food and not is_animal:
        return f"FOOD File uploaded and caption generated:<br><b>{caption_text}</b><br><a href='/'>Retour</a>"
    elif is_animal and is_food:
        return f"ANIMAL & FOOD File uploaded and caption generated:<br><b>{caption_text}</b><br><a href='/'>Retour</a>"
    else:
        return f"File uploaded and caption generated:<br><b>{caption_text}</b><br><a href='/'>Retour</a>"

@app.route("/images", methods=["GET"])
def show_images():
    images_with_captions = []
    for file in fs.find():
        data = file.read()
        encoded = base64.b64encode(data).decode('utf-8')
        caption_doc = captions_collection.find_one({"file_id": file._id})
        caption = caption_doc["caption"] if caption_doc else ""
        images_with_captions.append({"img": encoded, "caption": caption, "file_id": str(file._id)})

    return render_template_string("""
        <h2>Images Uploadées</h2>
        <form action="/delete_all" method="post" style="margin-bottom:20px;">
            <input type="submit" value="Tout supprimer" style="background-color:red;color:white;">
        </form>
        {% for item in images %}
            <img src="data:image/jpeg;base64,{{ item.img }}" style="max-width:300px; margin:10px;"><br>
            <p>{{ item.caption }}</p>
            <form action="/delete/{{ item.file_id }}" method="post" style="margin-bottom:20px;">
                <input type="submit" value="Supprimer">
            </form>
        {% endfor %}
        <a href="/">Retour à l'upload</a>
    """, images=images_with_captions)

@app.route("/delete_all", methods=["POST"])
def delete_all():
    # Supprimer toutes les images de GridFS
    for file in fs.find():
        fs.delete(file._id)
    # Supprimer toutes les captions
    captions_collection.delete_many({})
    return "Toutes les images et captions ont été supprimées.<br><a href='/images'>Retour</a>"

@app.route("/delete/<file_id>", methods=["POST"])
def delete(file_id):
    from bson import ObjectId
    fid = ObjectId(file_id)
    fs.delete(fid)
    captions_collection.delete_one({"file_id": fid})
    return "Image et caption supprimées avec succès"

# === Main ===
if __name__ == "__main__":
    app.run(debug=True)
