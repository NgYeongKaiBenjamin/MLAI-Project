import os
import shutil
import yaml
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_directory_structure(base_path):
    """Create the directory structure for processed data."""
    splits = ['train', 'valid', 'test']
    for split in splits:
        Path(base_path / split / 'images').mkdir(parents=True, exist_ok=True)
        Path(base_path / split / 'labels').mkdir(parents=True, exist_ok=True)

def validate_image(image_path):
    """Validate if an image can be opened and is not corrupted."""
    try:
        img = cv2.imread(str(image_path))
        return img is not None
    except Exception:
        return False

def get_split_data(source_base_path, split_name):
    """Get all valid images and their corresponding labels from a split directory."""
    valid_pairs = []
    split_path = source_base_path / split_name
    
    for class_name in os.listdir(split_path):
        class_path = split_path / class_name
        images_dir = class_path / 'images'
        labels_dir = class_path / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"Warning: Missing directory in {class_path}")
            continue
        
        for img_path in images_dir.glob('*.*'):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
                
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            # Validate both image and label exist
            if label_path.exists() and validate_image(img_path):
                valid_pairs.append((img_path, label_path, class_name))
            else:
                print(f"Warning: Invalid or missing pair - {img_path}")
    
    return valid_pairs

def copy_files(file_pairs, dest_base_path, split_name):
    """Copy image and label files to destination directory."""
    dest_img_dir = dest_base_path / split_name / 'images'
    dest_label_dir = dest_base_path / split_name / 'labels'
    
    for img_path, label_path, class_name in tqdm(file_pairs, desc=f"Copying {split_name} files"):
        # Create unique filenames using class name prefix
        unique_name = f"{class_name}_{img_path.name}"
        unique_label_name = f"{class_name}_{label_path.name}"
        
        # Copy files with unique names
        shutil.copy2(img_path, dest_img_dir / unique_name)
        shutil.copy2(label_path, dest_label_dir / unique_label_name)

def process_data(config):
    """Process the data according to configuration."""
    # Convert paths to Path objects
    source_path = Path(config['paths']['data_path'])
    processed_path = Path(config['paths']['output_base_path'])
    
    # Create directory structure
    create_directory_structure(processed_path)
    
    # Process each split
    splits = ['train', 'valid', 'test']
    for split in splits:
        print(f"\nProcessing {split} split...")
        file_pairs = get_split_data(source_path, split)
        
        if not file_pairs:
            print(f"No valid data found for {split} split")
            continue
            
        print(f"Found {len(file_pairs)} valid image-label pairs")
        copy_files(file_pairs, processed_path, split)

def verify_processed_data(config):
    """Verify the processed data structure."""
    processed_path = Path(config['paths']['output_base_path'])
    
    for split in ['train', 'valid', 'test']:
        img_path = processed_path / split / 'images'
        label_path = processed_path / split / 'labels'
        
        n_images = len(list(img_path.glob('*.*'))) if img_path.exists() else 0
        n_labels = len(list(label_path.glob('*.*'))) if label_path.exists() else 0
        
        print(f"\n{split.capitalize()} split:")
        print(f"Images: {n_images}")
        print(f"Labels: {n_labels}")
        
        # Count images per class
        class_counts = {}
        for img_file in img_path.glob('*.*'):
            class_name = img_file.name.split('_')[0]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print("Images per class:")
        for class_name, count in class_counts.items():
            print(f"{class_name}: {count}")

def main():
    """Main function to process the data."""
    print("Loading configuration...")
    config = load_config()
    
    print("\nStarting data processing...")
    process_data(config)
    
    print("\nVerifying processed data...")
    verify_processed_data(config)
    
    print("\nData processing completed successfully!")

if __name__ == "__main__":
    main()