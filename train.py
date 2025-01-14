from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
    """Memory-optimized data generator preparation with balanced class sizes."""
    image_size = config['image_size']
    train_path = config['paths']['train_data']
    valid_path = config['paths']['valid_data']
    batch_size = config['train']['batch_size']
    
    def preprocess_image(image):
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image_size, image_size))
        image = image.astype(np.float32) / 255.0
        return image

    # Create data augmentation pipeline
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Remove this if preprocessing already scales
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    valid_datagen = ImageDataGenerator(
        rescale=1./255  # Remove this if preprocessing already scales
    )

    def get_class_sizes(data_path):
        """Get the number of valid images per class after parsing labels."""
        class_sizes = {}
        
        for class_folder in os.listdir(data_path):
            if not os.path.isdir(os.path.join(data_path, class_folder)):
                continue
                
            images_folder = os.path.join(data_path, class_folder, "images")
            labels_folder = os.path.join(data_path, class_folder, "labels")
            
            if not (os.path.exists(images_folder) and os.path.exists(labels_folder)):
                continue
                
            valid_images = sum(1 for img_file in os.listdir(images_folder)
                             if img_file.lower().endswith(('.jpg', '.jpeg', '.png'))
                             and os.path.exists(os.path.join(labels_folder, 
                                              os.path.splitext(img_file)[0] + '.txt')))
            
            class_sizes[class_folder] = valid_images
            
        return class_sizes

    def create_generator(data_path, datagen, batch_size, is_training=True):
        class_sizes = get_class_sizes(data_path)
        min_class_size = min(class_sizes.values())
        
        while True:
            images = []
            labels = []
            count = 0
            
            class_folders = [f for f in os.listdir(data_path) 
                           if os.path.isdir(os.path.join(data_path, f))]
            
            for class_folder in class_folders:
                class_idx = config['classes'].index(class_folder)
                images_folder = os.path.join(data_path, class_folder, "images")
                
                if not os.path.exists(images_folder):
                    continue
                
                image_files = [f for f in os.listdir(images_folder)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                np.random.shuffle(image_files)
                
                for img_file in image_files[:min_class_size]:
                    if count == batch_size:
                        # Convert lists to numpy arrays
                        batch_images = np.array(images)
                        batch_labels = np.array(labels)
                        yield batch_images, batch_labels
                        images = []
                        labels = []
                        count = 0
                    
                    img_path = os.path.join(images_folder, img_file)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    # Convert BGR to RGB and resize
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (image_size, image_size))
                    
                    if is_training:
                        # Apply augmentation to individual image
                        img = img.astype(np.float32)  # Convert to float32
                        img = img.reshape((1,) + img.shape)  # Add batch dimension
                        img = next(datagen.flow(img, batch_size=1))[0]  # Get augmented image
                    
                    images.append(img)
                    labels.append(tf.keras.utils.to_categorical(class_idx, 
                                                             len(config['classes'])))
                    count += 1
            
            if images:
                batch_images = np.array(images)
                batch_labels = np.array(labels)
                yield batch_images, batch_labels

    def calculate_steps(path, batch_size):
        class_sizes = get_class_sizes(path)
        min_class_size = min(class_sizes.values())
        total_samples = min_class_size * len(class_sizes)
        return max(1, total_samples // batch_size)

    train_steps = calculate_steps(train_path, batch_size)
    valid_steps = calculate_steps(valid_path, batch_size)

    print(f"Training steps per epoch: {train_steps}")
    print(f"Validation steps per epoch: {valid_steps}")

    train_generator = create_generator(train_path, train_datagen, batch_size, is_training=True)
    valid_generator = create_generator(valid_path, valid_datagen, batch_size, is_training=False)

    return (train_generator, train_steps), (valid_generator, valid_steps), len(config['classes'])

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
    
    # Modify the build_model function to use fewer filters and add regularization:
    model = Sequential([
        # Input layer
        tf.keras.layers.InputLayer(input_shape=(image_size, image_size, 3)),
        
        # First block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Second block
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        
        # Third block
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),
        
        # Fourth block (added)
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),
        
        # Dense layers
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary with shape information
    model.summary()
    
    return model

def train_model(model, train_data, valid_data, config):
    """Memory-optimized training function with enhanced callbacks and data augmentation."""
    train_generator, train_steps = train_data
    valid_generator, valid_steps = valid_data

    # Create checkpoint directory in user space
    import os
    checkpoint_dir = config['paths']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model_temp.h5')
    final_model_path = os.path.join(config['paths']['saved_model'], 'final_model.h5')
    os.makedirs(config['paths']['saved_model'], exist_ok=True)

    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomFlip("horizontal")
    ])

    class SafeModelCheckpoint(ModelCheckpoint):
        """Custom checkpoint callback with safe saving mechanism."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.best_weights = None

        def on_epoch_end(self, epoch, logs=None):
            current = logs.get(self.monitor)
            if self.monitor_op(current, self.best):
                self.best = current
                self.best_weights = self.model.get_weights()
                print(f'\nEpoch {epoch + 1}: {self.monitor} improved from {self.best} to {current}')

    def scheduler(epoch, lr):
        """Custom learning rate scheduler function."""
        if epoch < 10:
            return lr  # Keep the initial learning rate for the first 10 epochs
        else:
            return lr * tf.math.exp(-0.1)  # Exponentially decay learning rate

    # Enhanced callbacks configuration
    callbacks = [
        # Existing callbacks
        SafeModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,  # Increased patience
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001  # Minimum change to qualify as an improvement
        ),
        # ReduceLROnPlateau(
        #     monitor='val_loss',
        #     factor=0.2,
        #     patience=5,
        #     min_lr=1e-6,
        #     verbose=1,
        #     cooldown=2  # Added cooldown period
        # ),
        # LearningRateScheduler(scheduler, verbose=1)
    ]

    try:
        tf.config.run_functions_eagerly(True)
        tf.get_logger().setLevel('ERROR')
        # Apply data augmentation to training data
        augmented_train_generator = (
            (data_augmentation(images, training=True), labels)
            for images, labels in train_generator
        )

        # Train the model
        history = model.fit(
            augmented_train_generator,
            steps_per_epoch=train_steps,
            validation_data=valid_generator,
            validation_steps=valid_steps,
            epochs=config['train']['epochs'],
            callbacks=callbacks,
            workers=0,  # Disable multiprocessing to reduce file handling issues
            use_multiprocessing=False
        )

        # Load best weights if checkpoint exists
        if os.path.exists(checkpoint_path):
            model.load_weights(checkpoint_path)
            print(f"Loaded weights from checkpoint: {checkpoint_path}")
        else:
            print(f"No checkpoint found at {checkpoint_path}, skipping weight loading.")

        # Save final model
        model.save(final_model_path)
        print(f"Model successfully saved to: {final_model_path}")
        test_model(model, valid_data, config)

        # Save model artifacts
        print("\nSaving model artifacts...")
        try:
            os.makedirs(os.path.dirname(final_model_path), exist_ok=True)

            # Save model architecture
            model_json = model.to_json()
            architecture_path = os.path.join(os.path.dirname(final_model_path), 'model_architecture.json')
            with open(architecture_path, 'w') as f:
                f.write(model_json)
            print(f"Saved model architecture to {architecture_path}")

            # Save weights
            weights_path = os.path.join(os.path.dirname(final_model_path), 'model_weights.h5')
            model.save_weights(weights_path)
            print(f"Saved model weights to {weights_path}")

            # Save training history
            history_path = os.path.join(os.path.dirname(final_model_path), 'training_history.npy')
            np.save(history_path, history.history)
            print(f"Saved training history to {history_path}")

        except Exception as save_error:
            print(f"Warning: Error saving model artifacts: {str(save_error)}")

        return history

    except Exception as train_error:
        print(f"Error during training: {str(train_error)}")
        raise
    
    finally:
        # Cleanup
        try:
            if os.path.exists(checkpoint_path):
                import shutil
                shutil.rmtree(checkpoint_path)
        except:
            print("Warning: Could not clean up temporary directory")



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

def test_model(model, valid_data, config):
    """Evaluate the trained model on validation data and display a classification report and confusion matrix."""
    valid_generator, valid_steps = valid_data
    class_labels = config['classes']

    # Evaluate the model
    print("\nEvaluating the model on validation data...")
    val_loss, val_accuracy = model.evaluate(valid_generator, steps=valid_steps, verbose=1)
    print(f"\nValidation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Generate predictions
    print("\nGenerating predictions...")
    y_true = []
    y_pred = []

    for i, (images, labels) in enumerate(valid_generator):
        if i >= valid_steps:
            break
        predictions = model.predict(images)
        y_true.extend(np.argmax(labels, axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))

    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

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