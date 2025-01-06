from tensorflow.keras.metrics import MeanSquaredError
import tensorflow as tf
import cv2
import sys
import numpy as np

# Load the trained CNN model from the .h5 file and specify custom metrics
cnn_model_path = "saved_model/model.h5"
cnn_model = tf.keras.models.load_model(cnn_model_path, custom_objects={'mse': MeanSquaredError()})

# Image size used in train.py
image_size = 128

# Initialize webcam
cap = cv2.VideoCapture(0)

if cap.isOpened():
    print("Camera OK")
else:
    cap.open()

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
    print(f"Predictions: {predictions}")

    # For demonstration, show predictions on the frame (modify as needed based on your output)
    annotated_frame = frame.copy()

    # Modify the display based on your output (adjust according to model type)
    label = f"Prediction: {np.argmax(predictions)}"  # If your model does classification
    # For bounding box or other detection models, adjust accordingly

    # Display the label on the frame
    cv2.putText(annotated_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with CNN predictions
    cv2.imshow("CNN Detection", annotated_frame)

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sys.exit()
