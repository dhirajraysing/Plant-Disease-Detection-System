import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

from main import categories


# Function to load and preprocess image
def load_image(image_file):
    img = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), 1)
    img = cv2.resize(img, (128, 128)) / 255.0
    return img

# Load trained model
model = tf.keras.models.load_model('sugarcane_leaf_disease_model.keras')

# Streamlit interface
st.title("Sugarcane Leaf Disease Detection")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = load_image(uploaded_file)
    st.image(img, channels="RGB", use_column_width=True)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    st.write(f"Predicted Disease: {categories[class_idx]}")