import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import yaml

def load_config(config_path):
    """Load configurations from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def prepare_data_generators(config):
    """Prepare data generators for training and validation."""
    image_size = config['image_size']
    batch_size = config['train']['batch_size']
    train_data_path = config['paths']['train_data']
    valid_data_path = config['paths']['valid_data']

    datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = datagen.flow_from_directory(
        train_data_path,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = datagen.flow_from_directory(
        valid_data_path,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, validation_generator

def build_model(image_size, num_classes):
    """Build and return the CNN model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Output layer for multi-class classification
    ])
    return model

def compile_model(model, config):
    """Compile the model with the given optimizer and loss function."""
    learning_rate = config['train']['learning_rate']
    loss_function = config['train']['loss_function']
    optimizer_name = config['train']['optimizer']

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) if optimizer_name == 'adam' else None
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

def train_and_save_model(model, train_generator, validation_generator, config):
    """Train the model and save it to the specified path."""
    epochs = config['train']['epochs']
    saved_model_path = config['paths']['saved_model']
    log_dir = config['paths']['logs']

    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator
    )

    # Save the trained model in .keras format
    model.save(saved_model_path)

    # Optional: Save training logs
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    return history

def test_model(model, test_generator):
    """Evaluate the model on the test dataset and return metrics."""
    print("Evaluating model on test data...")
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    return test_loss, test_accuracy

def plot_training_history(history):
    """Plot training and validation accuracy and loss."""
    # Accuracy plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
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
    # Load configurations
    config_path = 'dataset_source.yaml'
    config = load_config(config_path)

    # Prepare data generators
    train_generator, validation_generator = prepare_data_generators(config)

    # Number of classes (derived from YAML file)
    num_classes = len(config['classes'])

    # Build the model
    model = build_model(config['image_size'], num_classes)

    # Compile the model
    compile_model(model, config)

    # Train and save the model
    history = train_and_save_model(model, train_generator, validation_generator, config)

    # Load the test dataset
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_data_path = config['paths']['test_data']
    test_generator = test_datagen.flow_from_directory(
        test_data_path,
        target_size=(config['image_size'], config['image_size']),
        batch_size=config['train']['batch_size'],
        class_mode='categorical',
        shuffle=False
    )

    # Test the model
    test_loss, test_accuracy = test_model(model, test_generator)

    # Plot training history
    plot_training_history(history)

if __name__ == '__main__':
    main()
