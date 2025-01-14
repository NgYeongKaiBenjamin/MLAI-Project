import tensorflow as tf
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
from preprocessing_utils import preprocess_image  # Import the preprocessing function

# Load configuration
config_path = "config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

class_labels = config['classes']
image_size = config['image_size']

# Load model
cnn_model_path = f"{config['paths']['saved_model']}/final_model.h5"
try:
    cnn_model = tf.keras.models.load_model(cnn_model_path)
    print(f"Model loaded successfully from {cnn_model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    sys.exit(1)

plt.ion()
fig, ax = plt.subplots()
exit_flag = [False]

def on_close(event):
    exit_flag[0] = True

fig.canvas.mpl_connect('close_event', on_close)

# Add debug information
print(f"Model input shape: {cnn_model.input_shape}")
print(f"Model output shape: {cnn_model.output_shape}")
print(f"Class labels: {class_labels}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Preprocess using the same function as training
    try:
        processed_frame = preprocess_image(frame, image_size)
        # Add batch dimension
        model_input = np.expand_dims(processed_frame, axis=0)
        
        # Debug shapes
        print(f"Input shape: {model_input.shape}")
        
        # Make prediction
        predictions = cnn_model.predict(model_input, verbose=0)
        print(f"Raw predictions: {predictions}")
        
        detected_class_index = np.argmax(predictions[0])
        detected_class_label = class_labels[detected_class_index]
        confidence = predictions[0][detected_class_index]
        
        # Print all class probabilities for debugging
        for i, prob in enumerate(predictions[0]):
            print(f"{class_labels[i]}: {prob:.4f}")
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        break

    # Annotate frame
    annotated_frame = frame.copy()
    cv2.putText(
        annotated_frame,
        f"{detected_class_label} ({confidence:.2f})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display
    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    ax.clear()
    ax.imshow(rgb_frame)
    ax.set_title("Real-Time Detection")
    ax.axis("off")
    plt.pause(0.001)

    if exit_flag[0]:
        break

cap.release()
plt.close()