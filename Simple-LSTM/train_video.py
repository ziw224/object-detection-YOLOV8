import cv2
import numpy as np
import tensorflow as tf
from model import Model  # Make sure this is the correct import path for your model
from dataset import DataLoader  # Ensure DataLoader can handle video frame preprocessing if necessary

def build_model():
    # Define model architecture
    model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(None, 64, 64, 3)),  # Adjusted for image sequences
    tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu')),
    tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Dense(2)
])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')  # Using mean squared error loss and Adam optimizer

    return model

def preprocess_frame(frame):
    # Resize frame to expected input size of the model
    resized_frame = cv2.resize(frame, (64, 64))
    
    # Normalize pixel values if your model expects values between 0 and 1
    normalized_frame = resized_frame / 255.0
    
    return normalized_frame

def overlay_predictions(frame, prediction):
    # Assume prediction is scaled between 0 and 1
    # Scale predictions back to frame size
    original_size = frame.shape[1], frame.shape[0]  # Width and height
    x, y = prediction[0] * original_size[0], prediction[1] * original_size[1]
    
    # Convert coordinates to integers
    x, y = int(x), int(y)
    
    # Draw a circle at the prediction point
    # Arguments are: image, center coordinates, radius, color, thickness
    annotated_frame = cv2.circle(frame, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
    
    return annotated_frame

# Load or build your model
model = build_model()

# Load the video
cap = cv2.VideoCapture('./data/video/test_video.mp4')

# Define the codec and create a VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('./output/output_video.mp4', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame for your model
    # This step highly depends on your model's requirements
    preprocessed_frame = preprocess_frame(frame)
    
    # Predict using your model
    prediction = model.predict(np.array([preprocessed_frame]))  # Assuming model expects batched input
    
    # Overlay the predictions on the frame
    # This assumes `prediction` includes coordinates you can overlay; adjust as needed
    annotated_frame = overlay_predictions(frame, prediction)
    
    # Write the frame with predictions
    out.write(annotated_frame)

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
