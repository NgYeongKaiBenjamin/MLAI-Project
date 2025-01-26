import cv2
import numpy as np

def preprocess_image(image, target_size):
    """
    Preprocess image consistently for both training and inference.
    """
    # Convert BGR to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(image, (target_size, target_size))
    
    # Normalize to [0,1]
    image = image.astype(np.float32) / 255.0
    
    return image

def load_and_preprocess_image(image_path, target_size):
    """
    Load and preprocess image from file.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return preprocess_image(image, target_size)