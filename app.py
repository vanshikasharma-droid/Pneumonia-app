import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = load_model("model.h5")

st.title("🫁 Pneumonia Detection App")

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(128,128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)

    if prediction[0][0] > prediction[0][1]:
        st.error("🦠 Pneumonia Detected")
    else:
        st.success("✅ Normal")