import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# Load the trained Keras model
model = tf.keras.models.load_model('saved_model/best_model.keras')

# Image size used during training
image_size = 128  # Ensure this matches the image size used in training

st.write("""
         # Food Classification App
         """
         )

st.write("This app classifies images as either Eggtarts or Salmon Sashimi.")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is None:
    st.text("You haven't uploaded an image file")
else:
    # Load and display the uploaded image
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image (resize and normalize)
    image_resized = image.resize((image_size, image_size))
    image_array = np.array(image_resized) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Run the Keras model prediction
    predictions = model.predict(image_array)[0]  # Get the predictions for the first batch

    # Define the threshold for classification
    threshold = 0.5
    classes = ['Eggtarts', 'Salmon Sashimi']

    # Determine the predicted classes
    detected_classes = []
    for i, prob in enumerate(predictions):
        if prob >= threshold:
            detected_classes.append((classes[i], prob))

    # Display results
    if detected_classes:
        for cls, prob in detected_classes:
            st.write(f"Class: {cls}, Confidence: {prob:.2%}")
        st.success("Classification completed successfully!")
    else:
        st.warning("No class detected with sufficient confidence.")
