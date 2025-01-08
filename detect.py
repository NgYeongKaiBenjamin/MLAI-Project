from tensorflow.keras.metrics import MeanSquaredError
import tensorflow as tf
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

# Load the trained CNN model from the .h5 file and specify custom metrics
cnn_model_path = "saved_model/model.keras"
cnn_model = tf.keras.models.load_model(cnn_model_path, custom_objects={'mse': MeanSquaredError()})

# Image size used in train.py
image_size = 128

# Initialize webcam
cap = cv2.VideoCapture(0)

if cap.isOpened():
    print("Camera OK")
else:
    cap.open()

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

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image")
        break

    # Preprocess the frame for CNN (resize and normalize)
    frame_resized = cv2.resize(frame, (image_size, image_size)) / 255.0  # Normalize the frame
    frame_resized = np.expand_dims(frame_resized, axis=0)  # Add batch dimension

    # Make predictions using the CNN model
    predictions = cnn_model.predict(frame_resized)
    detected_class = np.argmax(predictions)  # 0 for no egg tart, 1 for egg tart
    print(f"Detected Class: {detected_class}")  # Output 0 or 1 to console

    # Annotate the frame based on predictions
    annotated_frame = frame.copy()
    label = "Egg Tart" if detected_class == 1 else "No Egg Tart"
    cv2.putText(annotated_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
        break

cap.release()
plt.close()  # Close the Matplotlib window
sys.exit()
