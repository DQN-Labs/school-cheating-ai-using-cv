import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import keras

# Load the trained model
model = keras.saving.load_model('image_classification_model.h5')  


class_indices = {
    0: 'cheat',
    1: 'not cheat'
}


def visualize_predictions(image_paths):
    plt.figure(figsize=(15, 10))  
    for i, img_path in enumerate(image_paths):
        # Load and preprocess the image
        test_image = image.load_img(img_path, target_size=(224, 224))  
        test_image = image.img_to_array(test_image) / 255.0 
        test_image = np.expand_dims(test_image, axis=0) 
        
        # Make prediction
        result = model.predict(test_image)
        predicted_index = np.argmax(result)  
        
        
        if predicted_index == 0:
            print(f"CHEATER DETECTED in {img_path}!")
        else:
            print(f"No cheating detected in {img_path}.")
        
        
        plt.subplot(3, 3, i + 1)  
        plt.imshow(test_image[0])
        plt.title(f'Predicted: {class_indices[predicted_index]}')
        plt.axis('off')  
    
    plt.show() 


image_paths = [
    '/workspaces/school-cheating-ai-using-cv/high-angle-kid-cheating-school-test.jpg',
    '/workspaces/school-cheating-ai-using-cv/cheating images for ai/cheating 2.jpg',
    '/workspaces/school-cheating-ai-using-cv/testing _images/not_cheat/test_3.jfif'

]

visualize_predictions(image_paths)
