import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
import keras
import serial

# Load the trained model
model = keras.saving.load_model('version_0-7.keras')

# Define class indices
class_indices = {
    0: 'cheat',
    1: 'not cheat'
}

# Initialize the serial communication
arduino = serial.Serial(port='COM3', baudrate=9600, timeout=1)  # Adjust port as needed

# Function to process the frame and predict
def process_frame(frame):
    # Resize and preprocess the frame
    frame_resized = cv2.resize(frame, (224, 224))
    frame_array = image.img_to_array(frame_resized) / 255.0
    frame_array = np.expand_dims(frame_array, axis=0)
    
    # Make prediction
    result = model.predict(frame_array)
    predicted_index = np.argmax(result)

    # Check for cheater
    if predicted_index == 0:  # 'cheat' class
        print("CHEATER DETECTED!")
        arduino.write(b'1')  # Send signal to Arduino
    else:
        arduino.write(b'0')  # Send non-cheating signal

    return class_indices[predicted_index]

# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Process the frame
    prediction = process_frame(frame)

    # Display the frame with the prediction
    cv2.putText(frame, f"Prediction: {prediction}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Cheating Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
arduino.close()
