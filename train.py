from ultralytics import YOLO
import torch

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load a YOLOv8 model (pretrained or custom)
    model = YOLO('yolov8n.pt')  # Use yolov8n.pt or your trained model checkpoint

    # Check GPU availability
    print("CUDA available:", torch.cuda.is_available())  # Should return True if GPU is available
    print("Current CUDA device:", torch.cuda.current_device())  # Prints the current GPU device number
    print("Device name:", torch.cuda.get_device_name(0))  # Prints the name of the GPU

    # Train the model on your custom dataset
    model.train(
        data='dataset_source.yaml',  # Path to the dataset config
        epochs=5,                    # Number of training epochs
        batch=16,                    # Batch size
        device=device,               # Device to use
        imgsz=640,                   # Image size
        plots=False                  # Disable plotting
    )

    # Save the trained model
    model.save("yolov8n_custom.pt")  # Save the model to a file

    # Evaluate the model on the validation set
    metrics = model.val()
    print(metrics)

    # Make predictions using the trained model
    results = model.predict(source='C:/YoloVenv/datasets/Data/test/images/image1Test.jpg', conf=0.25)

    # Process and display results
    for result in results:
        print(result)  # Print prediction details
        result.show()  # This will visualize the result if supported

if __name__ == "__main__":
    main()
