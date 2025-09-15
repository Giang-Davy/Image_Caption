#!/usr/bin/env python3
from flask import Blueprint, render_template_string
from config import fs, captions_collection
import base64
from bson import ObjectId

image_bp = Blueprint("images", __name__)

@image_bp.route("/images", methods=["GET"])
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

@image_bp.route("/delete_all", methods=["POST"])
def delete_all():
    for file in fs.find():
        fs.delete(file._id)
    captions_collection.delete_many({})
    return "Toutes les images et captions ont été supprimées.<br><a href='/images'>Retour</a>"

@image_bp.route("/delete/<file_id>", methods=["POST"])
def delete(file_id):
    fid = ObjectId(file_id)
    fs.delete(fid)
    captions_collection.delete_one({"file_id": fid})
    return "Image et caption supprimées avec succès"
