import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

def build_model():
    # Define the TensorFlow model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(64, 64, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2)  # Output: predicting x, y positions
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# def preprocess_frame(frame):
#     resized_frame = cv2.resize(frame, (64, 64))
#     normalized_frame = resized_frame / 255.0
#     return normalized_frame

# def overlay_predictions(frame, prediction):
#     original_size = frame.shape[1], frame.shape[0]
#     x, y = prediction[0] * original_size[0], prediction[1] * original_size[1]
#     annotated_frame = cv2.circle(frame, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)
#     return annotated_frame

def preprocess_frame(frame, x_min, y_min, x_max, y_max):
    cropped_frame = frame[y_min:y_max, x_min:x_max]
    resized_frame = cv2.resize(cropped_frame, (64, 64))  
    normalized_frame = resized_frame / 255.0  # Normalize 
    return np.expand_dims(normalized_frame, axis=0)  # Add batch dimension


tf_model = build_model() 

# Load object tracking information from CSV
object_tracking_info = pd.read_csv('./video_data/object_tracking_info.csv')

# Setup video input
cap = cv2.VideoCapture('./data/video/test_video.mp4')

# Dynamically get the frame size from the video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Setup video output with dynamic frame size
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('./output/annotated_output_video.mp4', fourcc, 20.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    objects_in_frame = object_tracking_info[object_tracking_info['frame'] == current_frame]

    for _, obj in objects_in_frame.iterrows():
        # Extract object bounding box coordinates
        x_min, y_min, x_max, y_max = obj['x_min'], obj['y_min'], obj['x_max'], obj['y_max']
        
        # Preprocess the object patch
        preprocessed_patch = preprocess_frame(frame, x_min, y_min, x_max, y_max)
        
        # Predict using TensorFlow model
        prediction = tf_model.predict(preprocessed_patch)[0]
        
        # Example annotation based on prediction (modify as needed)
        # Here, assume prediction is new object position, drawing a line to it
        new_x, new_y = int(prediction[0] * (x_max - x_min) + x_min), int(prediction[1] * (y_max - y_min) + y_min)
        cv2.arrowedLine(frame, (int((x_min+x_max)/2), int((y_min+y_max)/2)), (new_x, new_y), (0, 255, 0), 2)

    out.write(frame)  # Write the frame with annotations to the output video
    cv2.imshow('frame', frame)
    c = cv2.waitKey(1)
    if c & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print("Part 2 complete. Annotated video saved.")
