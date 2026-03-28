import streamlit as st
from PIL import Image
import torch

from predict import load_model, predict_image, get_priority
from utils.preprocess import get_transforms
from utils.gradcam import generate_gradcam, overlay_heatmap

# Load model
model = load_model()

st.title("Chest X-ray Triage System")

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Original Image", use_column_width=True)

    # Prediction
    score = predict_image(image, model)
    priority = get_priority(score)

    st.subheader("Prediction Results")
    st.write(f"Risk Score: {score:.2f}")
    st.write(f"Priority Level: {priority}")

    # Grad-CAM
    transform = get_transforms()
    image_tensor = transform(image).unsqueeze(0)

    cam = generate_gradcam(model, image_tensor)
    heatmap = overlay_heatmap(image, cam)

    st.subheader("Model Explanation (Grad-CAM)")
    st.image(heatmap, caption="Heatmap Overlay", use_column_width=True)