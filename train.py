import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import os
import yaml
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

def load_yolo_data(yaml_path, image_size):
    """
    Load YOLO-labeled data from the YAML dataset configuration.
    """
    with open(yaml_path, 'r') as file:
        dataset_config = yaml.safe_load(file)

    # Base path for datasets
    base_path = dataset_config['path']
    images_dir = os.path.join(base_path, dataset_config['train'])
    labels_dir = images_dir.replace('images', 'labels')  # Derive labels directory from images path

    images = []
    labels = []

    for label_file in os.listdir(labels_dir):
        label_path = os.path.join(labels_dir, label_file)
        image_path = os.path.join(images_dir, label_file.replace('.txt', '.jpg'))

        if os.path.exists(image_path):
            # Load and preprocess the image
            img = cv2.imread(image_path)
            img = cv2.resize(img, (image_size, image_size)) / 255.0
            images.append(img)

            # Load labels
            image_labels = []
            with open(label_path, 'r') as lf:
                for line in lf:
                    data = line.strip().split()
                    class_id = int(data[0])
                    bbox = list(map(float, data[1:]))  # YOLO format: [x_center, y_center, width, height]
                    image_labels.append([class_id] + bbox)

            labels.append(image_labels)

    return np.array(images), labels  # Keep labels as a list of lists


def preprocess_labels(labels, num_classes, image_size, max_boxes=50):
    """
    Convert YOLO bounding box labels into fixed-size tensors for training.
    Args:
        labels: List of YOLO-style bounding box labels.
        num_classes: Number of classes.
        image_size: Size of the input image (assumed square).
        max_boxes: Maximum number of boxes per image.

    Returns:
        Processed labels tensor (batch_size, max_boxes, num_classes + 4).
    """
    processed_labels = []

    for label in labels:
        boxes = []
        for bbox in label:
            class_id = bbox[0]
            bbox_coords = bbox[1:]
            # Normalize bbox coordinates to [0, 1] relative to image size
            bbox_coords = [coord / image_size for coord in bbox_coords]
            one_hot_class = [0] * num_classes
            one_hot_class[class_id] = 1
            boxes.append(one_hot_class + bbox_coords)

        if len(boxes) > max_boxes:
            boxes = boxes[:max_boxes]
        else:
            boxes += [[0] * (num_classes + 4)] * (max_boxes - len(boxes))

        processed_labels.append(boxes)

    return np.array(processed_labels, dtype=np.float32)


def create_cnn_model(input_shape, num_classes, max_boxes):
    """
    Build a CNN model for YOLO-style object detection.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(max_boxes * (num_classes + 4), activation='linear'),
        layers.Reshape((max_boxes, num_classes + 4))
    ])
    return model


def main():
    # Parameters
    yaml_path = 'dataset_source.yaml'
    image_size = 128  # Resized dimensions of input images
    batch_size = 16
    epochs = 200
    num_classes = 1  # Adjust based on your task (e.g., classification or object detection)
    max_boxes = 50  # Maximum number of bounding boxes per image
    model_save_path = "saved_model/model.keras"  # Save with .h5 extension

    # Check if a saved model exists
    if os.path.exists(model_save_path):
        print("Loading existing model...")
        model = load_model(model_save_path)  # Load the pre-trained model
    else:
        print("No existing model found. Training from scratch...")
        # Load YOLO-labeled data
        images, labels = load_yolo_data(yaml_path, image_size)
        print(f"Loaded {len(images)} images and labels.")

        # Preprocess labels
        labels = preprocess_labels(labels, num_classes, image_size, max_boxes)

        # Split data into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

        # Build CNN model
        input_shape = (image_size, image_size, 3)
        model = create_cnn_model(input_shape, num_classes, max_boxes)

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='mse',  # Use mean squared error for bounding box regression
                      metrics=['accuracy'])

        # Train the model
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))

        # Save the trained model
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")

    # Evaluate the model
    metrics = model.evaluate(x_val, y_val, verbose=2)
    print(f"Validation Metrics: {metrics}")

    # Predict on a test image
    test_image_path = 'C:/YoloVenv/datasets/Data/test/images/image1Test.jpg'
    test_img = cv2.imread(test_image_path)
    test_img_resized = cv2.resize(test_img, (image_size, image_size)) / 255.0
    test_img_resized = np.expand_dims(test_img_resized, axis=0)  # Add batch dimension

    predictions = model.predict(test_img_resized)
    print(f"Predictions: {predictions}")

if __name__ == "__main__":
    main()
