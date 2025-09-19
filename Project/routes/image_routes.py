#!/usr/bin/env python3
from flask import Blueprint, render_template, session, redirect
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

    return render_template("images.html", images=images_with_captions)


@image_bp.route("/delete_all", methods=["POST"])
def delete_all():
    if "user_id" not in session:
        return redirect("/login")

    user_id = session['user_id']
    for caption_doc in captions_collection.find({"user_id": user_id}):
        fs.delete(caption_doc["file_id"])
    captions_collection.delete_many({"user_id": user_id})

    return redirect("/images")


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

    return redirect("/images")
