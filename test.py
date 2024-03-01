# Import necessary libraries
from kalmanfilter import KalmanFilter
import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
from vidgear.gears import CamGear
from tracker import Tracker

# Initialize the YOLO model
model = YOLO('yolov8s.pt')

# Start video stream from a YouTube link
stream = CamGear(source='https://www.youtube.com/watch?v=ZvMZx-zCypM', stream_mode=True, logging=True).start()
# https://www.youtube.com/watch?v=ZvMZx-zCypM
# https://www.youtube.com/watch?v=Y1jTEyb3wiI
# https://www.youtube.com/watch?v=3vrnQ-MFOmY

# Setup OpenCV window
cv2.namedWindow('Frame')

# Load class list from coco.txt file
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Initialize Tracker
tracker = Tracker()

# Kalman Filter
kf = KalmanFilter()

# Main loop to process video frames
while True:
    frame = stream.read()

    # Resize frame for consistent processing
    frame = cv2.resize(frame, (1020, 500))
    
    # Predict objects in the frame using YOLO model
    results = model.predict(frame)
    boxes = results[0].boxes.data  # Extract bounding boxes data
    px = pd.DataFrame(boxes).astype("float")  # Convert to DataFrame for easier handling
    
    # List to store bounding boxes for tracking
    objects_rect = []
    for index, row in px.iterrows():
        x1, y1, x2, y2, d = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[5])
        objects_rect.append([x1, y1, x2-x1, y2-y1])  # Convert to the format expected by Tracker

    # Update tracker with detections
    objects_bbs_ids = tracker.update(objects_rect)
    
    for obj_bb_id in objects_bbs_ids:
        x, y, w, h, object_id = obj_bb_id
        cx = (x + w // 2)
        cy = (y + h // 2)

        predicted = kf.predict(cx, cy)  # Assuming this method returns a tuple/list

        # Draw bounding box and class name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cvzone.putTextRect(frame, f'{class_list[d]}', (x, y), 1, 1)

        # Draw predicted trajectory arrow if predicted is valid
        if predicted is not None and len(predicted) >= 2:
            cv2.arrowedLine(frame, (cx, cy), (int(predicted[0]), int(predicted[1])), (0, 255, 0), 2, tipLength=0.2)

    # Draw real trajectories
    tracker.draw_trajectories(frame)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cv2.destroyAllWindows()
stream.stop()