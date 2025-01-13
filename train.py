from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import yaml
import os
import tensorflow as tf
import numpy as np
import cv2
import sys

def check_gpu_availability():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # Set memory limit to 90% of available VRAM
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=5400)]  # 5.4GB for RTX 2060
            )
            print("GPU memory configuration set successfully.")
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")
    
    if tf.test.is_gpu_available():
        print("TensorFlow is using the GPU.")
    else:
        print("TensorFlow is using the CPU.")

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def normalize_class_name(name):
    name_mapping = {
        'Eggtarts': 'Eggtarts',
        'Salmon Sashimi':'Salmon Sashimi',
        'Unknown':'Unknown'
    }
    return name_mapping.get(name, name)

def preprocess_image(image, target_size):
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (target_size, target_size))
    image = image.astype(np.float32) / 255.0
    return image

def prepare_data_generators(config):
    """Memory-optimized data generator preparation."""
    image_size = config['image_size']
    train_path = config['paths']['train_data']
    valid_path = config['paths']['valid_data']
    batch_size = config['train']['batch_size']

    def data_generator(data_path, batch_size):
        while True:
            images = []
            labels = []
            count = 0

            class_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
            
            # Find the class with the smallest number of images
            min_class_size = float('inf')
            for class_folder in class_folders:
                images_folder = os.path.join(data_path, class_folder, "images")
                if os.path.exists(images_folder):
                    class_size = len([f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    min_class_size = min(min_class_size, class_size)

            for class_folder in class_folders:
                class_idx = config['classes'].index(class_folder)
                images_folder = os.path.join(data_path, class_folder, "images")

                if not os.path.exists(images_folder):
                    continue

                image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                # Shuffle image files and take the first min_class_size images
                np.random.shuffle(image_files)
                image_files = image_files[:min_class_size]

                for img_file in image_files:
                    if count == batch_size:
                        yield np.array(images), np.array(labels)
                        images = []
                        labels = []
                        count = 0

                    img_path = os.path.join(images_folder, img_file)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    img = preprocess_image(img, image_size)
                    images.append(img)
                    labels.append(tf.keras.utils.to_categorical(class_idx, len(config['classes'])))
                    count += 1

            if images:
                yield np.array(images), np.array(labels)

    def count_images(path):
        total = 0
        for class_folder in os.listdir(path):
            images_folder = os.path.join(path, class_folder, "images")
            if os.path.exists(images_folder):
                total += len([f for f in os.listdir(images_folder) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        return total

    train_steps = max(1, count_images(train_path) // batch_size)
    valid_steps = max(1, count_images(valid_path) // batch_size)

    print(f"Training steps per epoch: {train_steps}")
    print(f"Validation steps per epoch: {valid_steps}")

    return (data_generator(train_path, batch_size), train_steps), \
           (data_generator(valid_path, batch_size), valid_steps), \
           len(config['classes'])

def build_model(image_size, num_classes):
    """Memory-optimized model architecture with shape validation."""
    print(f"\nBuilding model with:")
    print(f"Input shape: ({image_size}, {image_size}, 3)")
    print(f"Number of classes: {num_classes}")
    
    # Validate parameters
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError(f"Invalid image_size: {image_size}")
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError(f"Invalid num_classes: {num_classes}")
    
    model = Sequential([
        # Input layer
        tf.keras.layers.InputLayer(input_shape=(image_size, image_size, 3)),
        
        # First block - start with fewer filters
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),
        
        # Third block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),
        
        # Dense layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary with shape information
    model.summary()
    
    return model

def train_model(model, train_data, valid_data, config):
    """Memory-optimized training function with improved error handling."""
    train_generator, train_steps = train_data
    valid_generator, valid_steps = valid_data
    
    # Create checkpoint directory if it doesn't exist
    try:
        checkpoint_dir = os.path.dirname(config['paths']['saved_model'])
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Using checkpoint directory: {checkpoint_dir}")
    except PermissionError:
        # Fallback to a user-writable directory
        import tempfile
        checkpoint_dir = tempfile.gettempdir()
        print(f"Permission denied for original directory. Using temporary directory: {checkpoint_dir}")
    
    model_path = os.path.join(checkpoint_dir, 'best_model.h5')
    final_model_path = os.path.join(checkpoint_dir, 'final_model.h5')
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    try:
        history = model.fit(
            train_generator,
            steps_per_epoch=train_steps,
            validation_data=valid_generator,
            validation_steps=valid_steps,
            epochs=config['train']['epochs'],
            callbacks=callbacks,
            workers=1,
            use_multiprocessing=False
        )
        
        # Try to save the final model
        try:
            model.save(final_model_path)
            print(f"Final model saved to: {final_model_path}")
        except Exception as e:
            print(f"Warning: Could not save final model to {final_model_path}")
            print(f"Error: {str(e)}")
            # Try saving to temporary directory
            temp_path = os.path.join(tempfile.gettempdir(), 'final_model.h5')
            model.save(temp_path)
            print(f"Model saved to temporary location: {temp_path}")
        
        return history
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

def plot_metrics(history, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    metrics = ['accuracy', 'loss']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for idx, metric in enumerate(metrics):
        axes[idx].plot(history.history[metric], label=f'Training {metric}')
        axes[idx].plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        axes[idx].set_title(f'{metric.capitalize()} Over Epochs')
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel(metric.capitalize())
        axes[idx].legend()
        axes[idx].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.close()

def evaluate_model(model, valid_data, class_labels, output_dir):
    """Evaluate model using the validation generator."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get predictions for a subset of validation data
    valid_generator, valid_steps = valid_data
    predictions = []
    true_labels = []
    
    for _ in range(valid_steps):
        batch_images, batch_labels = next(valid_generator)
        batch_predictions = model.predict(batch_images, verbose=0)
        predictions.extend(batch_predictions)
        true_labels.extend(batch_labels)
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    y_true = np.argmax(true_labels, axis=1)
    y_pred = np.argmax(predictions, axis=1)
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_labels)
    print("\nClassification Report:")
    print(report)
    
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    return report, cm

def save_model_summary(model, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(os.path.join(output_dir, 'model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

def create_directories(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directory created: {path}")
        else:
            print(f"Directory already exists: {path}")

def main():
    # Clear any existing sessions
    tf.keras.backend.clear_session()
    
    # Enable mixed precision training
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    config_path = 'config.yaml'
    config = load_config(config_path)
    
    # Ensure all necessary directories exist
    create_directories([
        config['paths']['saved_model'],
        config['paths']['logs'],
        config['paths']['output_dir'],
    ])
    
    check_gpu_availability()
    
    print("\nPreparing data generators...")
    train_data, valid_data, num_classes = prepare_data_generators(config)
    
    print("\nBuilding model...")
    model = build_model(config['image_size'], num_classes)
    save_model_summary(model, config['paths'].get('output_dir', 'output'))
    
    print("\nTraining model...")
    history = train_model(model, train_data, valid_data, config)
    
    print("\nPlotting metrics...")
    plot_metrics(history, config['paths'].get('plots', 'plots'))
    
    print("\nEvaluating model...")
    report, cm = evaluate_model(
        model,
        valid_data,
        config['classes'],
        config['paths'].get('output_dir', 'output')
    )
    
    # Save model architecture and weights
    model_dir = os.path.dirname(config['paths']['saved_model'])
    model_json = model.to_json()
    with open(os.path.join(model_dir, 'model_architecture.json'), 'w') as f:
        f.write(model_json)
    
    weights_path = os.path.join(model_dir, 'model_weights.h5')
    model.save_weights(weights_path)
    
    # Clear session
    tf.keras.backend.clear_session()
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: {config['paths']['saved_model']}")
    print(f"Logs saved to: {config['paths']['logs']}")
    print(f"Plots saved to: {config['paths'].get('plots', 'plots')}")

if __name__ == '__main__':
    main()