import pandas as pd
from transformers import CLIPTokenizer
import torch
import numpy as np
import os
import qai_hub
from PIL import Image


def process_image(image_path, target_size=(224, 224)):
    """Load and preprocess one image → (1, C, H, W) float32 in [0, 1]."""
    image = Image.open(image_path).convert("RGB").resize(target_size)
    image_array = np.array(image, dtype=np.float32) / 255.0
    return np.transpose(image_array, (2, 0, 1))[np.newaxis, :]  # (1, 3, 224, 224)


def load_images_from_folder(folder_path, target_size=(224, 224)):
    """Load and preprocess all images in a folder, sorted by filename."""
    image_paths = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".webp"))
    ])
    return [process_image(p, target_size) for p in image_paths]


# -------------------------------------------------------
# Images
# -------------------------------------------------------
# TODO: set to your image folder
image_folder = "dataset/images"

input_images = load_images_from_folder(image_folder)
print(f"Processed {len(input_images)} images — shape: {input_images[0].shape}")

image_dataset_id = qai_hub.upload_dataset({"image": input_images})
print(f"Image dataset ID: {image_dataset_id}")


# -------------------------------------------------------
# Text
# -------------------------------------------------------
# TODO: set to your text CSV (column 1 = text prompts)
csv_path = "dataset/txt_list.csv"
df = pd.read_csv(csv_path)
prompts = df.iloc[:, 1].dropna().tolist()

# Competition evaluation uses the ViT-B/32 CLIP tokenizer — matches our model
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

tokenized_texts = []
for prompt in prompts:
    tokens = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt"
    )["input_ids"].to(torch.int32)          # (1, 77)  int32
    tokenized_texts.append(tokens.numpy())

print(f"Tokenized {len(tokenized_texts)} prompts — shape: {tokenized_texts[0].shape}, dtype: {tokenized_texts[0].dtype}")

text_dataset_id = qai_hub.upload_dataset({"text": tokenized_texts})
print(f"Text dataset ID: {text_dataset_id}")
