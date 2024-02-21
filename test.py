# Import necessary libraries
import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
from vidgear.gears import CamGear
from tracker import Tracker

# Initialize the YOLO model
model = YOLO('yolov8s.pt')

# Start video stream from a YouTube link
stream = CamGear(source='https://www.youtube.com/watch?v=3vrnQ-MFOmY', stream_mode=True, logging=True).start()

# Callback function for mouse movements on the window
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse Position (BGR Colors): {x}, {y}")

# Setup OpenCV window and mouse callback
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Load class list from coco.txt file
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Initialize tracker
tracker = Tracker()

# Main loop variables
count = 0  # Frame counter to process every other frame

# Main loop to process video frames
while True:
    frame = stream.read()
    count += 1
    if count % 2 != 0:  # Skip every other frame to reduce processing
        continue

    # Resize frame for consistent processing
    frame = cv2.resize(frame, (1020, 500))
    
    # Predict objects in the frame using YOLO model
    results = model.predict(frame)
    boxes = results[0].boxes.data  # Extract bounding boxes data
    px = pd.DataFrame(boxes).astype("float")  # Convert to DataFrame for easier handling
    
    object_list = []  # List to store bounding boxes for tracking
    # Loop through detected objects and filter for objects
    for index, row in px.iterrows():
        x1, y1, x2, y2, d = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[5])
        class_name = class_list[d]
        if 'car' in class_name or 'person' in class_name:
            object_list.append([x1, y1, x2, y2])

    # Update tracker with detected objects
    bbox_idx = tracker.update(object_list)
    
    # Visualize detected objects and tracking
    for bbox in bbox_idx:
        x3, y3, x4, y4, id1 = bbox
        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 2)
        cvzone.putTextRect(frame, f'{class_list[d]}', (x3, y3), 1, 1)

    # Display the frame
    cv2.imshow("RGB", frame)

    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cv2.destroyAllWindows()
stream.stop()
