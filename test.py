# Import necessary libraries
from sklearn.linear_model import LinearRegression
import numpy as np
from kalmanfilter import KalmanFilter
import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
from vidgear.gears import CamGear
from tracker import Tracker

# Define a function to predict the future position using linear regression
def predict_future_position(trajectory, future_frames=10):
    if len(trajectory) < 2:
        return None  # Not enough data to predict future position

    # Prepare the data for linear regression
    past_positions = np.array(trajectory).reshape(-1, 2)
    times = np.arange(len(past_positions)).reshape(-1, 1)

    # Fit the linear regression model
    model_x = LinearRegression().fit(times, past_positions[:, 0])
    model_y = LinearRegression().fit(times, past_positions[:, 1])

    # Predict future position
    future_time = np.array([len(trajectory) + future_frames]).reshape(-1, 1)
    future_x = model_x.predict(future_time)[0]
    future_y = model_y.predict(future_time)[0]

    return int(future_x), int(future_y)

# Initialize the YOLO model
model = YOLO('yolov8s.pt')

# Start video stream from a YouTube link
stream = CamGear(source='https://www.youtube.com/watch?v=ZvMZx-zCypM', stream_mode=True, logging=True).start()
# https://www.youtube.com/watch?v=ZvMZx-zCypM highway
# https://www.youtube.com/watch?v=Y1jTEyb3wiI highway
# https://www.youtube.com/watch?v=3vrnQ-MFOmY
# https://www.youtube.com/watch?v=Obzq2YJ8uB8 pedestrians

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
        x1, y1, x2, y2, conf, d = int(row[0]), int(row[1]), int(row[2]), int(row[3]), float(row[4]), int(row[5])
        objects_rect.append([x1, y1, x2-x1, y2-y1, d])  # Convert to the format expected by Tracker

    # Update tracker with detections
    objects_bbs_ids = tracker.update(objects_rect)
    
    for obj_bb_id in objects_bbs_ids:
        x, y, w, h, object_id, class_id = obj_bb_id 

        # Draw bounding box and class name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        class_name = class_list[class_id]
        cvzone.putTextRect(frame, f'{class_name}', (x, y), 1, 1)

        # Get trajectory for the current object
        trajectory = tracker.trajectories[object_id].get_positions()

        # Predict the future position using the trajectory
        future_position = predict_future_position(trajectory)

        # Draw trajectory for the current object
        if object_id in tracker.trajectories:
            positions = tracker.trajectories[object_id].get_positions()
            # Draw a marker for each position in the trajectory
            for pos in positions:
                cv2.drawMarker(frame, pos, (255, 0, 0), markerType=cv2.MARKER_DIAMOND, markerSize=5)
            # # Optionally, draw lines between markers
            # for i in range(1, len(positions)):
            #     cv2.line(frame, positions[i - 1], positions[i], (0, 0, 255), 2)
        
        # Draw the predicted future position if available
        if future_position:
            cv2.drawMarker(frame, future_position, (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=10)
            #  draw a line from the last known position to the predicted future position
            if len(trajectory) > 0:
                cv2.line(frame, trajectory[-1], future_position, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cv2.destroyAllWindows()
stream.stop()