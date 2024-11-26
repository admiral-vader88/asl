import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the trained model
model = load_model('sign_language_model.h5')

# Define class labels
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box for the hand
            h, w, _ = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            # Extract the region of interest (ROI)
            hand_roi = frame[y_min:y_max, x_min:x_max]
            if hand_roi.size > 0:
                # Preprocess the ROI
                gray_roi = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
                resized_roi = cv2.resize(gray_roi, (28, 28))
                normalized_roi = resized_roi / 255.0
                reshaped_roi = np.expand_dims(normalized_roi, axis=(0, -1))  # Shape: (1, 28, 28, 1)

                # Predict
                predictions = model.predict(reshaped_roi)
                predicted_index = np.argmax(predictions)
                predicted_label = class_labels[predicted_index]
                confidence = predictions[0][predicted_index] * 100

                # Display results on the frame
                cv2.putText(frame, f'{predicted_label} ({confidence:.2f}%)', (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the output frame
    cv2.imshow('Sign Language Detection', frame)

    # Exit on pressing 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
