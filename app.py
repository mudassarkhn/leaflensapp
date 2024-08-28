import os
import json
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the working directory and model path
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, 'trained_model/plant_disease_prediction_model.h5')

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load the class indices
class_indices = json.load(open(os.path.join(working_dir, 'class_indices.json')))

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image):
    target_size = (224, 224)
    img = Image.open(image)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown")

    return jsonify({'prediction': predicted_class_name})

if __name__ == '__main__':
    app.run(debug=True)
