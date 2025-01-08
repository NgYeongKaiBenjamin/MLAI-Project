import tensorflow as tf
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml

# Load configuration from the YAML file
config_path = "dataset_source.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Extract class labels and image size from the config
class_labels = config['classes']
image_size = config['image_size']

# Load the trained CNN model
cnn_model_path = config['paths']['saved_model']
try:
    cnn_model = tf.keras.models.load_model(cnn_model_path, custom_objects={'mse': tf.keras.metrics.MeanSquaredError()})
    print(f"Model loaded successfully from {cnn_model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    sys.exit(1)
else:
    print("Camera initialized successfully.")

# Set up Matplotlib for displaying frames
plt.ion()  # Interactive mode for real-time updates
fig, ax = plt.subplots()
exit_flag = [False]  # Use a mutable object to handle loop control


def on_close(event):
    """
    Event handler to set the exit flag when the Matplotlib window is closed.
    """
    exit_flag[0] = True


fig.canvas.mpl_connect('close_event', on_close)  # Bind the close event

# Real-time detection loop
while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image from the webcam.")
        break

    # Preprocess the frame for the CNN model
    try:
        frame_resized = cv2.resize(frame, (image_size, image_size)) / 255.0  # Normalize the frame
        frame_resized = np.expand_dims(frame_resized, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        break

    # Make predictions
    try:
        predictions = cnn_model.predict(frame_resized)
        detected_class_index = np.argmax(predictions)
        detected_class_label = class_labels[detected_class_index]
    except Exception as e:
        print(f"Error during prediction: {e}")
        break

    print(f"Detected Class: {detected_class_label} (Confidence: {predictions[0][detected_class_index]:.2f})")

    # Annotate the frame based on predictions
    annotated_frame = frame.copy()
    cv2.putText(annotated_frame, detected_class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert the frame to RGB for Matplotlib
    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # Display the annotated frame using Matplotlib
    ax.clear()
    ax.imshow(rgb_frame)
    ax.set_title("Real-Time Detection")
    ax.axis("off")
    plt.pause(0.001)  # Small pause to update the figure

    # Exit if the Matplotlib window is closed
    if exit_flag[0]:
        print("Exiting detection loop.")
        break

# Cleanup resources
cap.release()
plt.close()  # Close the Matplotlib window
sys.exit()
