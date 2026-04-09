import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import sys

sys.path.append("src")

from models.model import get_model
from utils.gradcam import GradCAM

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_model():
    try:
        model = get_model()
        model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
        model.eval()
        return model
    except Exception as e:
        print(e)
        return None

model = load_model()

if model is None:
    st.info("⚠️ Model file not included. Please download or retrain.")
    st.stop()

# Target layer
target_layer = model.features.denseblock4
gradcam = GradCAM(model, target_layer)

# -------------------------------
# UI
# -------------------------------
st.title("🩺 Pneumonia Detection with Grad-CAM")

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # -------------------------------
    # Load Image (SAFE)
    # -------------------------------
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # -------------------------------
    # Transform
    # -------------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    input_tensor = transform(image).unsqueeze(0)

    # -------------------------------
    # Prediction
    # -------------------------------
    # Prediction
    try:
        with torch.no_grad():
            output = model(input_tensor)

            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)

            class_names = ["Normal", "Pneumonia"]

            st.subheader(f"Prediction: {class_names[pred.item()]}")
            st.write(f"Confidence: {conf.item()*100:.2f}%")

    except Exception as e:
        st.error(f"Prediction Error: {e}")

    # -------------------------------
    # Grad-CAM (FIXED)
    # -------------------------------
    try:
        cam = gradcam.generate(input_tensor, pred.item())
    except Exception as e:
        st.error(f"Grad-CAM Error: {e}")
        st.stop()
    # Convert tensor → numpy safely
    if torch.is_tensor(cam):
        cam = cam.detach().cpu().numpy()

    cam = np.squeeze(cam)

    # Normalize
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # Convert original image
    img_cv = np.array(image.resize((224, 224)).convert("RGB"))
    img_cv = img_cv.astype(np.float32) / 255.0

    heatmap = heatmap.astype(np.float32) / 255.0

    # Superimpose
    superimposed = heatmap * 0.5 + img_cv * 0.5

    # -------------------------------
    # Show Grad-CAM
    # -------------------------------
    st.image(superimposed, caption="Grad-CAM", use_column_width=True)
    
