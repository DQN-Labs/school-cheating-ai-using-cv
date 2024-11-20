import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import keras
import serial
import time

# Load the trained model
model = keras.saving.load_model('version_0-7.keras')

# Define class indices
class_indices = {
    0: 'cheat',
    1: 'not cheat'
}

# Initialize serial communication with Arduino
arduino = serial.Serial(port='COM6', baudrate=9600, timeout=1)  # Replace 'COM3' with the correct port
time.sleep(2)  # Wait for the connection to initialize


# Function to send commands to Arduino
def send_command_to_arduino(command):
    try:
        arduino.write(command.encode())  # Send the command as bytes
        print(f"Sent to Arduino: {command}")
    except Exception as e:
        print(f"Error sending command: {e}")


# Function to visualize predictions
def visualize_predictions(image_paths):
    plt.figure(figsize=(15, 10))
    for i, img_path in enumerate(image_paths):
        test_image = image.load_img(img_path, target_size=(224, 224))
        test_image = image.img_to_array(test_image) / 255.0
        test_image = np.expand_dims(test_image, axis=0)

        # Get prediction from the model
        result = model.predict(test_image)
        predicted_index = np.argmax(result)

        # Send a signal to Arduino if a cheater is detected
        if predicted_index == 0:  # If 'cheat' is predicted
            print("CHEATER DETECTED!")
            send_command_to_arduino('1')  # Send '1' to Arduino
        else:
            send_command_to_arduino('0')  # Send '0' to Arduino

        # Visualize the image and prediction
        plt.subplot(3, 3, i + 1)
        plt.imshow(test_image[0])
        plt.title(f'Predicted: {class_indices[predicted_index]}')
        plt.axis('off')
    plt.show()


# List of image paths
image_paths = [
    "C:\\Users\srika\PycharmProjects\PythonProject\school-cheating-ai-using-cv-main\school-cheating-ai-using-cv-main\\new_data_copy\cheat\cheating_2.jpg",
    "C:\\Users\srika\PycharmProjects\PythonProject\school-cheating-ai-using-cv-main\school-cheating-ai-using-cv-main\\new_data_copy\cheat\IMG_6500_jpg.rf.4df62bbc2427279432344d28537ae78a.jpg   ",
    "C:\\Users\srika\PycharmProjects\PythonProject\school-cheating-ai-using-cv-main\school-cheating-ai-using-cv-main\\new_data_copy\\not_cheat\good928_jpg.rf.11c3beee95af291a6c2f7ff67fa07df9.jpg",
    "C:\\Users\srika\PycharmProjects\PythonProject\school-cheating-ai-using-cv-main\school-cheating-ai-using-cv-main\\new_data_copy\cheat\cheat1033_jpg.rf.5c58fac7a80c557522e2bfe55d358571.jpg"
]

# Run the visualization function
try:
    visualize_predictions(image_paths)
except KeyboardInterrupt:
    print("Program interrupted.")
finally:
    arduino.close()  # Close the serial connection
