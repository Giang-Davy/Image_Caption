#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import pandas as pd
import os

# Chemins vers les données
metadata_path = "HAM10000/HAM10000_metadata.csv"
images_path = "HAM10000/HAM10000_images"  # dossier fusionné

# Chargement du fichier de métadonnées
metadata = pd.read_csv(metadata_path)

# Conversion des labels en chaînes
metadata['dx'] = metadata['dx'].astype(str)

# Ajout du chemin complet pour chaque image
metadata['path'] = metadata['image_id'].apply(lambda x: os.path.join(images_path, f"{x}.jpg"))

# Division train/validation
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(metadata, test_size=0.2, stratify=metadata['dx'], random_state=42)

# Générateurs d’images avec augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_dataframe(
    train_df,
    x_col='path',
    y_col='dx',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_dataframe(
    val_df,
    x_col='path',
    y_col='dx',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Chargement du modèle EfficientNetB0 sans la couche de classification finale
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Gel des poids du modèle de base
base_model.trainable = False

# Ajout des couches de classification
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
num_classes = len(train_gen.class_indices)
output = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Compilation du modèle
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entraînement du modèle
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5
)

# Sauvegarde du modèle
model.save("efficientnetb0_ham10000.h5")
print("Modèle entraîné et sauvegardé sous 'efficientnetb0_ham10000.h5'")
