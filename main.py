import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Vegetable Prediction", layout="centered")

st.title("🥕 Vegetable Prediction System")
st.write("Capture an image of a vegetable to predict its class.")

# Load trained model
model = load_model("vegetable_model.keras")

# IMPORTANT: order must match training
class_names = ['Cabbage', 'Capsicum', 'Carrot', 'Potato', 'Tomato']

# Camera input
img_file = st.camera_input("Take a picture")

if img_file is not None:
    image = Image.open(img_file)
    img = np.array(image)

    # Preprocess image
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    preds = model.predict(img)
    idx = np.argmax(preds)
    confidence = preds[0][idx] * 100

    if confidence > 80:
        st.success(f"Prediction: {class_names[idx]} ({confidence:.2f}%)")
    else:
        st.warning("Low confidence. Try again with better lighting.")
