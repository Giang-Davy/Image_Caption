#!/usr/bin/env python3
from flask import Blueprint, request, render_template_string
from PIL import Image
import torch
from config import fs, captions_collection, food_collection, device
from models.caption_model import model, tokenizer, transform
from utils.caption_utils import contains_animal, contains_food

upload_bp = Blueprint("upload", __name__)

@upload_bp.route("/", methods=["GET"])
def index():
    return render_template_string("""
        <h2>Upload Image</h2>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="submit" value="Upload">
        </form>
        <a href="/images">Voir toutes les images</a>
    """)

@upload_bp.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files or request.files["file"].filename == "":
        return "No file uploaded", 400
    file = request.files["file"]
    file_id = fs.put(file, filename=file.filename)
    grid_out = fs.get(file_id)
    image = Image.open(grid_out).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    # Génération caption
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
    captions_collection.insert_one({"file_id": file_id, "caption": caption_text})
    is_animal = contains_animal(caption_text)
    is_food = contains_food(caption_text)
    if is_food:
        food_collection.insert_one({"file_id": file_id, "caption": caption_text, "is_food": True})
    if is_animal and not is_food:
        return f"ANIMAL File uploaded and caption generated:<br><b>{caption_text}</b><br><a href='/'>Retour</a>"
    elif is_food and not is_animal:
        return f"FOOD File uploaded and caption generated:<br><b>{caption_text}</b><br><a href='/'>Retour</a>"
    elif is_animal and is_food:
        return f"ANIMAL & FOOD File uploaded and caption generated:<br><b>{caption_text}</b><br><a href='/'>Retour</a>"
    else:
        return f"File uploaded and caption generated:<br><b>{caption_text}</b><br><a href='/'>Retour</a>"
