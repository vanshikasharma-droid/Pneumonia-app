import streamlit as st
import numpy as np
from PIL import Image

st.title("🫁 Pneumonia Detection App")

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Dummy result (for now)
    st.success("Prediction working ✅ (Dummy Result)")
