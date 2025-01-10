from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, TensorBoard
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import yaml
import os
import tensorflow as tf
import numpy as np

def check_gpu_availability():
    """Check and enable GPU availability."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled.")
        except RuntimeError as e:
            print(f"Error enabling GPU memory growth: {e}")
    
    if tf.test.is_gpu_available():
        print("TensorFlow is using the GPU.")
    else:
        print("TensorFlow is not using the GPU.")

# Verify GPU usage
check_gpu_availability()

def load_config(config_path):
    """Load configurations from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def prepare_data_generators(config):
    """Prepare data generators for training and validation."""
    image_size = config['image_size']
    batch_size = config['train']['batch_size']

    train_path = config['paths']['train_data']
    valid_path = config['paths']['valid_data']
    test_path = config['paths']['test_data']

    datagen_train = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    datagen_valid_test = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = datagen_train.flow_from_directory(
        train_path,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = datagen_valid_test.flow_from_directory(
        valid_path,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = datagen_valid_test.flow_from_directory(
        test_path,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator

def build_model(image_size, num_classes):
    """Build and return the CNN model with L2 regularization in the Dense layer."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def compile_model(model, config):
    """Compile the model."""
    learning_rate = config['train']['learning_rate']
    optimizer_name = config['train']['optimizer']
    loss_function = config['train']['loss_function']

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) if optimizer_name == 'adam' else None
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

def train_and_save_model(model, train_generator, validation_generator, config):
    """Train the model and save it."""
    epochs = config['train']['epochs']
    saved_model_path = config['paths']['saved_model']
    log_dir = config['paths']['logs']

    # Early stopping and learning rate scheduler
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = LearningRateScheduler(lambda epoch: config['train']['learning_rate'] * 0.95 ** epoch)

    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[early_stopping, lr_scheduler, TensorBoard(log_dir=log_dir)]
    )

    model.save(saved_model_path)
    print("Model has been saved")
    return history

def test_model(model, test_generator, class_labels):
    """Evaluate the model and print metrics."""
    print("Evaluating model on test data...")
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    y_true = test_generator.classes
    y_pred = np.argmax(model.predict(test_generator), axis=-1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    return test_loss, test_accuracy

def plot_training_history(history):
    """Plot training and validation accuracy and loss."""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    config_path = 'config.yaml'
    config = load_config(config_path)

    class_labels = config['classes']
    train_generator, validation_generator, test_generator = prepare_data_generators(config)
    num_classes = len(class_labels)

    with tf.device('/GPU:0'):  # Specify GPU usage
        model = build_model(config['image_size'], num_classes)
        compile_model(model, config)

        history = train_and_save_model(model, train_generator, validation_generator, config)
        test_model(model, test_generator, class_labels)
        plot_training_history(history)

if __name__ == '__main__':
    main()
