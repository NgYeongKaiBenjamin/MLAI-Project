import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Load YOLOv8 model

st.write("""
         # Object Detection with YOLOv8
         """
         )

st.write("This is a simple image classification web app to detect objects in an image using YOLOv8.")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is None:
    st.text("You haven't uploaded an image file")
else:
    # Load and display the uploaded image
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert PIL image to NumPy array
    image_np = np.array(image)

    # Run YOLOv8 prediction
    results = model.predict(source=image_np, save=False, show=False, conf=0.25)

    # Extract predictions
    detections = results[0].boxes.data.cpu().numpy()  # Get detection data

    st.write("Detected objects:")

    if len(detections) > 0:
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection  # Coordinates, confidence, class
            st.write(f"Class: {model.names[int(cls)]}, Confidence: {conf:.2f}")
    else:
        st.write("No objects detected!")
