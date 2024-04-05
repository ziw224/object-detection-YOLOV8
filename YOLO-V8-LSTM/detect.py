from collections import defaultdict
import cv2
from ultralytics import YOLO
import numpy as np
import csv
import os

# Initialize YOLO v8 model
yolo_model = YOLO('yolov8n.pt')  # Make sure this is the correct path to your model

# Video input setup
cap = cv2.VideoCapture('./data/video/test_video.mp4')  # Adjust path as necessary

# Prepare to record object tracking info
object_info = []
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"Processing frame {frame_count}...")

    # Run YOLOv8 tracking on the frame
    results = yolo_model.track(frame, persist=True)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        # Map class IDs to their corresponding class names
        class_names = [yolo_model.names[int(cid)] for cid in class_ids]

        # Iterate over each detected object
        for (box, track_id, class_name) in zip(boxes, track_ids, class_names):
            x_min, y_min, x_max, y_max = box[:4]
            # Now include frame_count in the object_info append
            object_info.append([frame_count, track_id, class_name, x_min, x_max, y_min, y_max])

# Cleanup
cap.release()

# Ensure the output directory exists
output_directory = './video_data'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)  # This creates the output directory 

# Path for the CSV file
csv_file_path = os.path.join(output_directory, 'object_tracking_info.csv')
with open(csv_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    # Include 'frame' in the header
    writer.writerow(['frame', 'object_id', 'class_name', 'x_min', 'x_max', 'y_min', 'y_max'])
    writer.writerows(object_info)

print("Part 1 complete. Object tracking info saved to CSV.")
print(f"Tracking information saved to {csv_file_path}.")
