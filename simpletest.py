from collections import defaultdict

import cv2
import numpy as np
from vidgear.gears import CamGear
from ultralytics import YOLO
from sklearn.linear_model import LinearRegression

# Define a function to predict the future position using linear regression
def predict_future_position(trajectory, future_frames=10):

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


# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "./sample.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])
stream = CamGear(source='https://www.youtube.com/watch?v=ZvMZx-zCypM', stream_mode=True, logging=True).start()

# Loop through the video frames
while True:
    frame = stream.read()
    # Resize frame for consistent processing
    frame = cv2.resize(frame, (1020, 500))

    

    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    # results = model.track(frame, persist=True)
    results = model.track(frame, persist=True)
    # print(results)
    boxes = []
    track_ids = []
    if results[0].boxes.id != None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        # confidences = results[0].boxes.conf.cpu().numpy().astype(int)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Plot the tracks
    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))  # x, y center point
        if len(track) > 30:  # retain 90 tracks for 90 frames
            track.pop(0)
        
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 0, 255), thickness=2)

        future_position = predict_future_position(track)

        if future_position:
            last_known_position = track[-1]  # Retrieve the last known position
            last_known_position = (int(last_known_position[0]), int(last_known_position[1]))
            future_position_int = (int(future_position[0]), int(future_position[1]))
            cv2.line(annotated_frame, last_known_position, future_position_int, (0, 255, 0), 2)

    # Display the frame with all annotations
    cv2.imshow("YOLOv8 Tracking", annotated_frame) 

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Proper cleanup
cv2.destroyAllWindows()
stream.stop()  # Use stream.stop() for CamGear cleanup