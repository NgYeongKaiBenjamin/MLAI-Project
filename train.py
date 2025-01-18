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
import math
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers.schedules import ExponentialDecay
    

def check_gpu_availability():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        print("No GPU available")
        return

    try:
        # Set memory growth before any virtual device configurations
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPUs:", gpus)
    except ValueError as e:
        print("Error enabling memory growth:", e)


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def prepare_data_generators(config):
    """Memory-optimized data generator preparation with class balancing."""
    image_size = config['image_size']
    train_path = config['paths']['train_data']
    valid_path = config['paths']['valid_data']
    batch_size = config['train']['batch_size']

    class BalancedDirectoryIterator(tf.keras.preprocessing.image.DirectoryIterator):
        def _set_filepaths_and_classes(self, directory):
            self.filenames = []
            self.classes = []
            classes = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
            self.num_classes = len(classes)
            self.class_indices = dict(zip(classes, range(len(classes))))
            
            # First pass: count images per class
            class_counts = {}
            for class_name in classes:
                images_path = os.path.join(directory, class_name, 'images')
                if os.path.exists(images_path):
                    class_counts[class_name] = len([f for f in os.listdir(images_path) 
                                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            # Find the size of the smallest class
            min_class_size = min(class_counts.values())
            print(f"\nBalancing classes to {min_class_size} images each (smallest class size)")
            
            # Second pass: limit each class to min_class_size
            for class_name in classes:
                class_idx = self.class_indices[class_name]
                images_path = os.path.join(directory, class_name, 'images')
                
                if os.path.exists(images_path):
                    image_files = [f for f in os.listdir(images_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    # Randomly select min_class_size images
                    selected_files = np.random.choice(image_files, min_class_size, replace=False)
                    
                    for fname in selected_files:
                        self.filenames.append(os.path.join(class_name, 'images', fname))
                        self.classes.append(class_idx)
            
            self.samples = len(self.filenames)
            print(f"Total samples after balancing: {self.samples}")

    # Create data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        validation_split=0.1
    )

    # Simple rescaling for validation
    valid_datagen = ImageDataGenerator(
        rescale=1./255
    )

    def print_class_distribution(data_path):
        print(f"\nClass distribution in {data_path}:")
        for class_name in os.listdir(data_path):
            class_path = os.path.join(data_path, class_name)
            if not os.path.isdir(class_path):
                continue
                
            images_dir = os.path.join(class_path, "images")
            if os.path.exists(images_dir):
                image_count = len([f for f in os.listdir(images_dir) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"- {class_name}: {image_count} images")

    print("\nOriginal class distribution:")
    print_class_distribution(train_path)
    print_class_distribution(valid_path)

    try:
        train_generator = BalancedDirectoryIterator(
            train_path,
            train_datagen,
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True,
            interpolation='nearest'
        )

        valid_generator = BalancedDirectoryIterator(
            valid_path,
            valid_datagen,
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False,
            interpolation='nearest'
        )

        train_steps = math.ceil(train_generator.samples / train_generator.batch_size)
        valid_steps = math.ceil(valid_generator.samples / valid_generator.batch_size)

        print(f"\nAfter balancing:")
        print(f"Training samples per class: {train_generator.samples // len(train_generator.class_indices)}")
        print(f"Validation samples per class: {valid_generator.samples // len(valid_generator.class_indices)}")
        print(f"Class mapping: {train_generator.class_indices}")
        print(f"Training steps per epoch: {train_steps}")
        print(f"Validation steps per epoch: {valid_steps}")

        return (train_generator, train_steps), (valid_generator, valid_steps), len(train_generator.class_indices)

    except Exception as e:
        print(f"Error setting up data generators: {str(e)}")
        raise

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

        # First block - basic features
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Second block - more complex features
        Conv2D(48, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(48, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Third block - high-level features
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(64, activation='relu',  # Increase from 48
            kernel_regularizer=tf.keras.regularizers.l2(0.003)),
        BatchNormalization(),
        Dropout(0.3),  # Reduce dropout
        Dense(num_classes, activation='softmax')
    ])
    

    # Use it in your optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary with shape information
    model.summary()
    
    return model

def train_model(model, train_data, valid_data, config):
    """Kaggle-optimized training function."""
    train_generator, train_steps = train_data
    valid_generator, valid_steps = valid_data

    # Create checkpoint directory
    checkpoint_dir = config['paths']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model_temp.h5')
    final_model_path = os.path.join(config['paths']['saved_model'], 'final_model.h5')
    os.makedirs(config['paths']['saved_model'], exist_ok=True)

    # Simplified callbacks to reduce memory usage
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            mode='min',  # Explicitly set mode
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            mode='min',  # Explicitly set mode
            patience=20,  # Increase patience since improvement is steady
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            mode='min',  # Explicitly set mode
            factor=0.2,
            patience=7,
            min_lr=1e-6,
            verbose=1
        )
    ]

    try:
        # Clear any existing session
        tf.keras.backend.clear_session()
        
        # Enable memory growth
        check_gpu_availability()
        
        # Reduce precision to save memory
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Verify data shapes before training
        print("\nVerifying data shapes:")
        sample_batch = next(iter(train_generator))
        print(f"Sample batch shapes - X: {sample_batch[0].shape}, y: {sample_batch[1].shape}")
        
        # Start training with garbage collection
        import gc
        class_weights = {
            0: 1.0,  # Eggtarts
            1: 1.2,  # Salmon Sashimi (slightly higher weight)
            2: 1.0   # Unknown
        }
        print("\nStarting training...")
        history = model.fit(
            train_generator,
            steps_per_epoch=train_steps,
            validation_data=valid_generator,
            validation_steps=valid_steps,
            epochs=config['train']['epochs'],
            callbacks=callbacks,
            verbose=1, 
            class_weight=class_weights
            # workers=2,  # Reduced number of workers
            # max_queue_size=10,  # Reduced queue size
            # use_multiprocessing=False  # Disable multiprocessing
        )
        
        gc.collect()  # Force garbage collection
        
        # Save final model
        print("\nSaving model...")
        model.save(final_model_path, save_format='h5')
        
        return history

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
            except Exception as e:
                print(f"Warning: Could not clean up checkpoint: {str(e)}")


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
    
    weights_path = os.path.join(model_dir, 'model_weights.keras')
    model.save_weights(weights_path)
    
    # Clear session
    tf.keras.backend.clear_session()
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: {config['paths']['saved_model']}")
    print(f"Logs saved to: {config['paths']['logs']}")
    print(f"Plots saved to: {config['paths'].get('plots', 'plots')}")

if __name__ == '__main__':
    main()