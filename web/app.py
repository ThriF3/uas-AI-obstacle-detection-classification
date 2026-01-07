from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import load_model as keras_load_model
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Model Configuration
IMG_SIZE = 224
NUM_CLASSES = 4
CLASS_NAMES = ['Mobil', 'Motor', 'Orang', 'Truk']

# Load and configure model
def load_model():
    # Load the complete .keras model file
    model = keras_load_model('./model/best_resnet50_model.keras') # Create a folder called "model" Then put your model there. Make sure the name format is the same
    
    return model

# Initialize model
try:
    model = load_model()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocess_frame(image_data):
    """Preprocess image for ResNet50"""
    try:
        # Decode base64 image
        img_data = base64.b64decode(image_data.split(',')[1])
        img = Image.open(BytesIO(img_data))
        img = img.convert('RGB')
        
        # Resize to model input size
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img)
        
        # Normalize (ResNet50 preprocessing)
        img_array = img_array.astype('float32')
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle frame prediction"""
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        })
    
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({
                'success': False,
                'error': 'No image data received'
            })
        
        # Preprocess image
        processed_img = preprocess_frame(image_data)
        
        if processed_img is None:
            return jsonify({
                'success': False,
                'error': 'Image preprocessing failed'
            })
        
        # Make prediction
        predictions = model.predict(processed_img, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        # Get all class probabilities
        class_probabilities = {
            CLASS_NAMES[i]: float(predictions[0][i]) 
            for i in range(NUM_CLASSES)
        }
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': class_probabilities
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)