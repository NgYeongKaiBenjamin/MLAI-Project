import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# Load the trained Keras model
model = tf.keras.models.load_model('saved_model/final_model.h5')

# Image size used during training
image_size = 128  # Ensure this matches the image size used in training

st.write("""
         # Food Classification App
         """
         )

st.write("This app classifies images as either Eggtarts or Salmon Sashimi, or Unknown if confidence is low.")

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
    threshold = 0.3
    classes = ['Eggtarts', 'Salmon Sashimi', 'Unknown']

    # Get the predicted class with the highest probability
    predicted_class_index = np.argmax(predictions)
    predicted_class = classes[predicted_class_index]
    confidence = predictions[predicted_class_index]

    # Check if the confidence is above the threshold
    if confidence >= threshold:
        st.write(f"Predicted Class: {predicted_class}")
        st.write(f"Confidence: {confidence:.2%}")
        st.success("Classification completed successfully!")
    else:
        # If confidence is below threshold, classify as Unknown
        st.write(f"Predicted Class: {classes[2]}")
        st.write(f"Confidence: {confidence:.2%}")
        st.warning("The model couldn't classify the image with sufficient confidence. It is classified as Unknown.")
