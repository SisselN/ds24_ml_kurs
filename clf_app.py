import streamlit as st
import os
import pickle
import numpy as np

from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from PIL import Image, ImageFilter, ImageOps, ImageEnhance

# Hämtar modellen från Huggingface-repository.
load_dotenv()
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
model_path = hf_hub_download(
    repo_id = "SisselN/voting_clf_MNIST",
    filename = "voting_model.pkl",
    use_auth_token = hf_api_key
)

with open(model_path, 'rb') as f:
    model = pickle.load(f)

threshold = 180
contrast_factor = 1.55

# Funktion för att förbehandla bild.
def preprocessing(image, threshold, contrast_factor):
    grayscale_img = image.convert('L')
    inverted_img = ImageOps.invert(grayscale_img)

    enhancer = ImageEnhance.Contrast(inverted_img)
    contrast_img = enhancer.enhance(contrast_factor)

    binarized_img = contrast_img.point(lambda x: 255 if x > threshold else 0)
    
    binarized_img = binarized_img.filter(ImageFilter.SHARPEN)

    # Beskär bilden.
    bbox = binarized_img.getbbox()
    if bbox:
        cropped_img = binarized_img.crop(bbox)
    else:
        cropped_img = Image.new('L', (20, 20), 255)

    # Skala om innan padding
    scale = 20 / max(cropped_img.size)
    new_size = (int(cropped_img.size[0] * scale), int(cropped_img.size[1] * scale))
    resized_cropped = cropped_img.resize(new_size, Image.BICUBIC)

    # Tillämpa skärpning efter omformning
    resized_cropped = resized_cropped.filter(ImageFilter.SHARPEN)
    
    # Lägger till en kant.
    border_size = max(2, (28 - max(resized_cropped.size)) // 2)
    padded_img = ImageOps.expand(resized_cropped, border=border_size, fill=255)

    new_img = Image.new('L', (28, 28), 255)
    left = (28 - padded_img.size[0]) // 2
    top = (28 - padded_img.size[1]) // 2
    new_img.paste(padded_img, (left, top))

    # Konverterar till rätt format.
    img_array = np.array(new_img, dtype=np.float32) / 255.0
    img_vector = img_array.flatten().reshape(1, -1)

    return img_vector

st.title("Maskininlärningsmodell som identifierar siffror.")

camera_image = st.camera_input("Håll upp en bild med en siffra på och klicka på 'Take photo'.")

if camera_image:
    img = Image.open(camera_image)
    prepped_img = preprocessing(img, threshold, contrast_factor)
    prediction = model.predict(prepped_img)

    st.success(f'Modellen säger att det är siffran {prediction} på bilden.')
