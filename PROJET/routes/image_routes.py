#!/usr/bin/env python3
from flask import Blueprint, render_template_string, session, redirect
from config import fs, captions_collection
import base64
from bson import ObjectId

image_bp = Blueprint("images", __name__)

@image_bp.route("/images", methods=["GET"])
def show_images():
    if "user_id" not in session:
        return redirect("/login")

    user_id = session['user_id']
    images_with_captions = []

    # On récupère uniquement les captions de l'utilisateur connecté
    for caption_doc in captions_collection.find({"user_id": user_id}):
        file_id = caption_doc["file_id"]
        try:
            file = fs.get(file_id)
            data = file.read()
            encoded = base64.b64encode(data).decode('utf-8')
            caption = caption_doc.get("caption", "")
            images_with_captions.append({
                "img": encoded,
                "caption": caption,
                "file_id": str(file_id)
            })
        except Exception:
            continue

    return render_template_string("""
        <h2>Mes Images Uploadées</h2>
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
        <br>
        <a href="/logout">Se déconnecter</a>
    """, images=images_with_captions)


@image_bp.route("/delete_all", methods=["POST"])
def delete_all():
    if "user_id" not in session:
        return redirect("/login")

    user_id = session['user_id']
    for caption_doc in captions_collection.find({"user_id": user_id}):
        fs.delete(caption_doc["file_id"])
    captions_collection.delete_many({"user_id": user_id})
    return "Toutes vos images et captions ont été supprimées.<br><a href='/images'>Retour</a>"


@image_bp.route("/delete/<file_id>", methods=["POST"])
def delete(file_id):
    if "user_id" not in session:
        return redirect("/login")

    user_id = session['user_id']
    fid = ObjectId(file_id)

    captions_collection.delete_one({"file_id": fid, "user_id": user_id})
    try:
        fs.delete(fid)
    except Exception:
        pass

    return "Image et caption supprimées avec succès<br><a href='/images'>Retour</a>"
