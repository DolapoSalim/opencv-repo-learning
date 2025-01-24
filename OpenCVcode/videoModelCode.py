import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the YOLOv8 model
model_path = "model/path"  # Update with your model path
model = YOLO(model_path)

# Path to the input video
video_path = "video/path"  # Update with your video path
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the video parameters
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Prepare to save the output video
output_video_path = "video/output/path"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec for MP4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (OpenCV uses BGR by default)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run inference
    results = model(img)

    # Get the bounding boxes and labels
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box in xyxy format
        confidence = box.conf[0].item()  # Confidence score
        class_id = int(box.cls[0].item())  # Class ID
        class_name = results[0].names[class_id]  # Class label
        
        # Draw the bounding box on the frame
        frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Annotate the frame with the class name and confidence
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Write the frame with annotations to the output video
    out.write(frame)

    # Optionally, display the frame (for testing)
    # cv2.imshow('Frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved to {output_video_path}")
