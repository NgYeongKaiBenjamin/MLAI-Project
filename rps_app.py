import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

# Load the trained Keras model (replace with your path)
model = tf.keras.models.load_model('saved_model/model.keras')

# Image size used during training
image_size = 128  # Make sure this matches the image size used in training

st.write("""
         # Object Classification with Keras CNN Model
         """
         )

st.write("This is a simple image classification web app to detect objects using a pre-trained Keras CNN model.")

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
    predictions = model.predict(image_array)

    # Get the class with the highest probability
    detected_class = np.argmax(predictions)

    # Class names as defined in your dataset
    class_names = ['no_eggtart', 'eggtart']  # Modify according to your actual class names

    st.write("Detected Class:")
    st.write(f"{class_names[detected_class]} with confidence: {predictions[0][detected_class]:.2f}")

    # Annotate the image with the predicted label
    annotated_image = image.copy()
    label = class_names[detected_class]
    annotated_image = np.array(annotated_image)

    # Convert the image back to RGB and display
    annotated_image = Image.fromarray(annotated_image)
    st.image(annotated_image, caption=f"Predicted: {label}", use_column_width=True)
