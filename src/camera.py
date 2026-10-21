"""Streamlit's native camera_input keeps webcam capture dependency-free."""
from PIL import Image
import streamlit as st

def capture_image():
    photo = st.camera_input("Take a photo", help="Permite el acceso a la cámara en el navegador si se solicita.")
    if photo is None:
        return None
    try:
        return Image.open(photo).convert("RGB")
    except Exception as exc:
        st.error(f"No se pudo leer la captura: {exc}")
        return None
