import os
import numpy as np
import cv2
import yaml
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import shutil

class DataGenerator:
    def __init__(self, config):
        self.image_size = config['image_size']
        self.batch_size = config['train']['batch_size']
        self.classes = config['classes']
        
    def create_generator(self, data_path):
        """Create a data generator for the given path."""
        def generator():
            while True:
                images = []
                labels = []
                count = 0
                
                class_folders = self.classes
                for class_idx, class_name in enumerate(class_folders):
                    class_path = os.path.join(data_path, class_name)
                    images_folder = os.path.join(class_path, "images")
                    
                    if not os.path.exists(images_folder):
                        continue
                        
                    image_files = [f for f in os.listdir(images_folder) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    np.random.shuffle(image_files)
                    
                    for img_file in image_files:
                        if count == self.batch_size:
                            yield np.array(images), np.array(labels)
                            images = []
                            labels = []
                            count = 0
                            
                        img_path = os.path.join(images_folder, img_file)
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                            
                        img = self.preprocess_image(img)
                        images.append(img)
                        labels.append(tf.keras.utils.to_categorical(class_idx, len(self.classes)))
                        count += 1
                        
                if images:
                    yield np.array(images), np.array(labels)
                    
        return generator()
    
    def preprocess_image(self, image):
        """Preprocess a single image."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.astype(np.float32) / 255.0
        return image
        
    def count_samples(self, data_path):
        """Count total number of samples in a directory."""
        total = 0
        for class_name in self.classes:
            images_folder = os.path.join(data_path, class_name, "images")
            if os.path.exists(images_folder):
                total += len([f for f in os.listdir(images_folder) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        return total

def prepare_data_generators(config):
    """Prepare data generators for training, validation, and testing."""
    data_gen = DataGenerator(config)
    
    # Use the correct paths from the config
    train_path = os.path.join(config['paths']['output_base_path'], 'train')
    valid_path = os.path.join(config['paths']['output_base_path'], 'valid')
    test_path = os.path.join(config['paths']['output_base_path'], 'test')
    
    train_generator = data_gen.create_generator(train_path)
    valid_generator = data_gen.create_generator(valid_path)
    test_generator = data_gen.create_generator(test_path)
    
    train_steps = max(1, data_gen.count_samples(train_path) 
                     // config['train']['batch_size'])
    valid_steps = max(1, data_gen.count_samples(valid_path) 
                     // config['train']['batch_size'])
    test_steps = max(1, data_gen.count_samples(test_path) 
                    // config['train']['batch_size'])
    
    return ((train_generator, train_steps), 
            (valid_generator, valid_steps), 
            (test_generator, test_steps),
            len(config['classes']))

def build_model(image_size, num_classes):
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', padding='same', 
               input_shape=(image_size, image_size, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_data, valid_data, config):
    """Train the model with the specified configuration."""
    train_generator, train_steps = train_data
    valid_generator, valid_steps = valid_data
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            config['paths']['saved_model'],
            monitor='val_loss',
            save_best_only=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        validation_data=valid_generator,
        validation_steps=valid_steps,
        epochs=config['train']['epochs'],
        callbacks=callbacks
    )
    
    return history

def plot_metrics(history, plots_path):
    """Plot and save training metrics."""
    metrics = ['accuracy', 'loss']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history[metric], label=f'Training {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.title(f'{metric.capitalize()} Over Time')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_path, f'{metric}_plot.png'))
        plt.close()

def main():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create necessary directories
    os.makedirs(config['paths']['saved_model'], exist_ok=True)
    os.makedirs(config['paths']['logs'], exist_ok=True)
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    os.makedirs(config['paths']['plots'], exist_ok=True)
    
    # Prepare data generators
    print("Preparing data generators...")
    data = prepare_data_generators(config)
    train_data, valid_data, test_data, num_classes = data
    
    # Build and train model
    print("Building model...")
    model = build_model(config['image_size'], num_classes)
    
    print("Training model...")
    history = train_model(model, train_data, valid_data, config)
    
    # Plot training metrics
    print("Plotting metrics...")
    plot_metrics(history, config['paths']['plots'])
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()