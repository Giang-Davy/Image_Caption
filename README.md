# Image Captioning Project

**📌 Description**

This project implements an image captioning system based on the Show-and-Tell architecture, fine-tuned on the COCO dataset.

The model generates descriptive captions for images using a pre-trained GPT-2 tokenizer and a ResNet50 encoder. It can produce both general image captions and food-specific captions.

Key features:

- Fine-tuning Show-and-Tell model with COCO images and captions.

- Caption generation for new images using the fine-tuned model.

- Works with both general and domain-specific objects (e.g., food).

**⚙️ Installation**

1. **Clone the repository**
```bash
git clone https://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@github.com/Giang-Davy/Portfolio.git
cd Portfolio
```

2. **Create a virtual environment**
```bash
python -m venv venv
```

Activate the environment:
```bash
# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

**📥 COCO Dataset**

Download the COCO dataset from kaggle.com : https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset and place images and annotations in PROJET/data/COCO.

**Training**
Run the training script trainGPT2.py to fine-tune the Show-and-Tell model with the dataset COCO:

```bash
python trainGPT2.py
```

**🖼️ Caption Generation**

To generate captions for a new image, run in /test:

```bash
python caption.py
```

Update the image path in caption.py:
```bash
image_path = "images/image.jpg"
```

**🛢️ MySQL Setup**

The web interface requires a MySQL server.
Install MySQL and create a database for the app with this link : https://dev.mysql.com/downloads/mysql/8.4.html

**🌐 Web Interface**

You can run the web interface using app.py to upload images and generate captions interactively.
```bash
python PROJET/app.py
```

Then open your browser and go to:
```bash
http://127.0.0.1:5000
```

**Infomartion**

This project was made by Dang-Vu Davy for the final exam of Holberton Shool and the certification RNCP 6 - Application Developer Designer

My email : davy.dangvu@gmail.com