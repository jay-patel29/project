import streamlit as st
import numpy as np
import cv2
import joblib
from streamlit_drawable_canvas import st_canvas

# Load model
model = joblib.load("model.pkl")

st.title("Handwritten Digit Recognition")

canvas = st_canvas(
    fill_color="black",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
)

if st.button("Predict"):
    if canvas.image_data is not None:
        img = canvas.image_data

        img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28))
        img = 255 - img
        img = img / 255.0

        img = img.reshape(1, -1)   # IMPORTANT for SVM/KNN

        pred = model.predict(img)

        st.success(f"Predicted Digit: {pred[0]}")
