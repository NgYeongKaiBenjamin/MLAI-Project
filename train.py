from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, SpatialDropout2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import yaml
import os
import tensorflow as tf
import numpy as np
import math
import gc
    
#Checks GPU availability
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

#Loads yaml file
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

#Prepares Data Generators
def prepare_data_generators(config):
    image_size = config['image_size']
    train_path = config['paths']['train_data']
    valid_path = config['paths']['valid_data']
    test_path = config['paths']['test_data']
    batch_size = config['train']['batch_size']

    def print_class_distribution(data_path): #Prints number of images per class
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
        channel_shift_range=50.0
    )

    # Simple rescaling for validation
    valid_datagen = ImageDataGenerator(
        rescale=1./255
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    #Create Data Generator
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        interpolation='nearest'
    )
    valid_generator = valid_datagen.flow_from_directory(
        valid_path,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        interpolation='nearest'
    )
    test_generator = test_datagen.flow_from_directory(  
        test_path,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        interpolation='nearest'
    )

    #Calculate Steps
    train_steps = math.ceil(train_generator.samples / train_generator.batch_size)
    valid_steps = math.ceil(valid_generator.samples / valid_generator.batch_size)
    test_steps = math.ceil(test_generator.samples / test_generator.batch_size)

    print(f"\nTraining samples: {train_generator.samples}")
    print(f"Validation samples: {valid_generator.samples}")
    print(f"Class mapping: {train_generator.class_indices}")
    print(f"Training steps per epoch: {train_steps}")
    print(f"Validation steps per epoch: {valid_steps}")

    return (train_generator, train_steps), (valid_generator, valid_steps), (test_generator, test_steps), len(train_generator.class_indices)

#Build model
def build_model(config,image_size, num_classes):
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

        #First Block
        Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2=0.0005)),
        Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2=0.0005)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        SpatialDropout2D(0.1), 

        # Second block with residual-like connection
        Conv2D(48, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2=0.0005)),
        Conv2D(48, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2=0.0005)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        SpatialDropout2D(0.1),

        # Third block with increased capacity
        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2=0.0005)),
        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2=0.0005)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        SpatialDropout2D(0.1),

        GlobalAveragePooling2D(), 

        # Memory-efficient classification head
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=0.0005)),
        BatchNormalization(),
        Dropout(0.2),  
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=0.0005)),
        BatchNormalization(),
        Dropout(0.2),

        Dense(num_classes, activation='softmax')
    ])
    #Optimizer using Adam
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0001,
        clipnorm=1.0
    )
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    #Compile Model
    model.compile(
        optimizer=optimizer,
        loss=config['train']['loss_function'],
        metrics=['accuracy']
    )
    
    # Print model summary with shape information
    model.summary()
    
    return model

#Learning Rate Scheduler
def get_learning_rate_scheduler(config):
    initial_lr = config['train'].get('learning_rate', 0.0001)
    min_lr = initial_lr * 0.01  # Minimum LR will be 1% of initial LR
    warmup_epochs = 5  # Warmup for first 5 epochs
    total_epochs = config['train']['epochs']
    
    def scheduler(epoch):
        # Warmup phase
        if epoch < warmup_epochs:
            return initial_lr * ((epoch + 1) / warmup_epochs)
        
        # Cosine decay phase
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + (initial_lr - min_lr) * cosine_decay

    return LearningRateScheduler(scheduler, verbose=1)

def train_model(model, train_data, valid_data, config):
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
            mode='min', 
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            mode='min', 
            patience=5,  
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            mode='min', 
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        get_learning_rate_scheduler(config)
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
        
        #Add class weightage to prevent class bias
        class_weights = {
            0: 1.06,  # Eggtarts
            1: 1.43,  # Salmon Sashimi 
            2: 0.74   # Unknown
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
        )
        
        gc.collect()  # Force garbage collection
        
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

#Plot metrics
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

def test_model(model, test_data, config):
    test_generator, test_steps = test_data
    class_labels = config['classes']

    print("\nEvaluating the model on test data...")
    test_loss, test_accuracy = model.evaluate(test_generator, steps=test_steps, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Generate predictions on test data
    y_true = []
    y_pred = []

    for i, (images, labels) in enumerate(test_generator):
        if i >= test_steps:
            break
        predictions = model.predict(images)
        y_true.extend(np.argmax(labels, axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))

    # Classification report and confusion matrix
    report = classification_report(y_true, y_pred, target_names=class_labels)
    print("\nTest Classification Report:")
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    print("\nTest Confusion Matrix:")
    print(cm)

    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Test Data Confusion Matrix')
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
    test_output_dir = config['paths'].get('output_diAr', 'output')
    os.makedirs(test_output_dir, exist_ok=True)
    plt.savefig(os.path.join(test_output_dir, 'test_confusion_matrix.png'))
    plt.close()

    # Save test classification report
    with open(os.path.join(test_output_dir, 'test_classification_report.txt'), 'w') as f:
        f.write(report)

    return report, cm

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
    train_data, valid_data, test_data, num_classes = prepare_data_generators(config)
    
    print("\nBuilding model...")
    model = build_model(config,config['image_size'], num_classes)
    save_model_summary(model, config['paths'].get('output_dir', 'output'))
    
    print("\nTraining model...")
    history = train_model(model, train_data, valid_data, config)
    
    print("\nPlotting metrics...")
    plot_metrics(history, config['paths'].get('plots', 'plots'))
    
    print("\nEvaluating model on validation data...")
    evaluate_model(
        model,
        valid_data,
        config['classes'],
        config['paths'].get('output_dir', 'output')
    )
    
    print("\nTesting model on test data...")
    test_report, test_cm = test_model(
        model,
        test_data,
        config
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