import mysql.connector

# Connexion à MySQL
def get_connection():
    conn = mysql.connector.connect(
        host="localhost",
        user="ton_utilisateur",       # remplace par ton utilisateur MySQL
        password="ton_motdepasse",    # remplace par ton mot de passe MySQL
        database="user_db"            # nom de la base que tu as créée
    )
    return conn
