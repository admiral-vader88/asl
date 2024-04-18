import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# Set TensorFlow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the pre-trained model
model = load_model('smnist.h5')

# Setup MediaPipe Hands
mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("Failed to grab frame")
    cap.release()
    raise SystemExit("Failed to grab initial frame from the camera. Exiting...")

# Get the dimensions of the frame
h, w, _ = frame.shape

# Define letters corresponding to model's classes
letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks

    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_min, x_max, y_min, y_max = w, 0, h, 0
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, x_max = min(x_min, x), max(x_max, x)
                y_min, y_max = min(y_min, y), max(y_max, y)
            y_min, y_max = max(y_min - 20, 0), min(y_max + 20, h)
            x_min, x_max = max(x_min - 20, 0), min(x_max + 20, w)

            # Extract ROI and preprocess it for the model
            roi = frame[y_min:y_max, x_min:x_max]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_resized = cv2.resize(roi_gray, (28, 28))
            roi_normalized = roi_resized / 255.0
            roi_expanded = np.expand_dims(roi_normalized, axis=[0, -1])  # Shape: (1, 28, 28, 1)

            # Predict using the pre-trained model
            predictions = model.predict(roi_expanded)
            predicted_index = np.argmax(predictions)
            predicted_letter = letterpred[predicted_index]
            confidence = predictions[0][predicted_index]

            # Display predicted letter and confidence on the frame
            cv2.putText(frame, f'{predicted_letter} - {confidence*100:.2f}%', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw rectangle around the hand
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

# Release the capture and destroy any OpenCV windows
cap.release()
cv2.destroyAllWindows()
