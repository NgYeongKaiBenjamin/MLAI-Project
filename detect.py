from ultralytics import YOLO
import cv2
import sys

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace with the path to your YOLOv8 model if necessary

# Initialize webcam
cap = cv2.VideoCapture(0)

if cap.isOpened():
    print("Camera OK")
else:
    cap.open()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image")
        break

    # YOLOv8 Prediction
    results = model.predict(frame, conf=0.25)  # Adjust confidence threshold as needed

    # Draw bounding boxes and labels on the frame
    annotated_frame = results[0].plot()

    # Display the frame with YOLOv8 predictions
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sys.exit()
