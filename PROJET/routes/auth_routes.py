# routes/auth_routes.py
#!/usr/bin/env python3
from flask import Blueprint, request, render_template_string, redirect, session
from db.mysql import get_connection
import hashlib
import mysql.connector

auth_bp = Blueprint("auth_bp", __name__)

# Page d'inscription (GET = formulaire, POST = création)
@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        email = (request.form.get("email") or "").strip()
        password = request.form.get("password") or ""

        if not username or not email or not password:
            return render_template_string("""
                <h2>Inscription</h2>
                <p style="color:red;">Tous les champs sont requis.</p>
                <a href="/register">Retour</a>
            """), 400

        # Calculer le hash du mot de passe (SHA256 ici pour rester compatible avec ton setup)
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        conn = get_connection()
        cursor = conn.cursor()
        try:
            # Insérer le hash dans la colonne password_hash
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                (username, email, password_hash)
            )
            conn.commit()
        except mysql.connector.IntegrityError:
            # username ou email déjà existant
            return render_template_string("""
                <h2>Inscription</h2>
                <p style="color:red;">Nom d'utilisateur ou email déjà utilisé.</p>
                <a href="/register">Retour</a>
            """), 400
        finally:
            cursor.close()
            conn.close()

        # Rediriger vers la page de connexion après inscription réussie
        return redirect("/login")

    # GET -> afficher le formulaire
    return render_template_string("""
        <h2>Inscription</h2>
        <form action="/register" method="post">
            <input type="text" name="username" placeholder="Nom d'utilisateur" required><br><br>
            <input type="email" name="email" placeholder="Email" required><br><br>
            <input type="password" name="password" placeholder="Mot de passe" required><br><br>
            <input type="submit" value="S'inscrire">
        </form>
        <p>Déjà inscrit ? <a href="/login">Se connecter</a></p>
    """)


# Page de connexion (GET = formulaire, POST = vérif)
@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        if not username or not password:
            return render_template_string("""
                <h2>Connexion</h2>
                <p style="color:red;">Tous les champs sont requis.</p>
                <a href="/login">Retour</a>
            """), 400

        password_hash = hashlib.sha256(password.encode()).hexdigest()

        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute(
                "SELECT id, username, email FROM users WHERE username=%s AND password_hash=%s",
                (username, password_hash)
            )
            user = cursor.fetchone()
        finally:
            cursor.close()
            conn.close()

        if user:
            # Stocker la session
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect("/")  # ou "/" selon ton app
        else:
            return render_template_string("""
                <h2>Connexion</h2>
                <p style="color:red;">Nom d'utilisateur ou mot de passe incorrect.</p>
                <a href="/login">Réessayer</a>
            """), 401

    # GET -> afficher le formulaire
    return render_template_string("""
        <h2>Connexion</h2>
        <form action="/login" method="post">
            <input type="text" name="username" placeholder="Nom d'utilisateur" required><br><br>
            <input type="password" name="password" placeholder="Mot de passe" required><br><br>
            <input type="submit" value="Se connecter">
        </form>
        <p>Pas encore inscrit ? <a href="/register">S'inscrire</a></p>
    """)


# Déconnexion
@auth_bp.route("/logout")
def logout():
    session.clear()
    return redirect("/")
