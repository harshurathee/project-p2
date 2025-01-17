# project-p2
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained model
model = load_model('plant_disease_model.h5')

# Function to preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# Predict disease
def predict_disease(image_path):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    disease_class = np.argmax(prediction)
    return disease_class

# Example usage
if __name__ == "__main__":
    image_path = 'leaf_sample.jpg'
    disease = predict_disease(image_path)
    print(f"Detected Disease: {disease}")
