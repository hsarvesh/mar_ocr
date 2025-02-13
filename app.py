from flask import Flask, request, jsonify, send_from_directory
import pytesseract
from PIL import Image
import cv2
import numpy as np
import os
import base64
import io

app = Flask(__name__)

def save_image(image, original_filename, step):
    """Helper function to save the image with a step description."""
    filename = f"{os.path.splitext(original_filename)[0]}_{step}.jpg"
    cv2.imwrite(filename, image)
    print(f"Saved image as {filename}")

def preprocess_image(image_data):
    """Loads and preprocesses the image for better OCR results."""
    try:
        # Decode base64 image data
        decoded_data = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(decoded_data))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        save_image(gray, "preprocessed", "grayscale")

        # Additional preprocessing steps can be added here
        # For example, thresholding, noise removal, etc.

        return gray
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        return None

def extract_text(image):
    try:
        pytesseract.pytesseract.tesseract_cmd = 'D:\\Tesseract-OCR\\tesseract.exe'  # Or the full path on Windows
        custom_config = r'--oem 3 --psm 1 -l mar+eng' # Adjust PSM if needed
        text = pytesseract.image_to_string(image, config=custom_config)
        return text
    except pytesseract.TesseractNotFoundError:
        return "Error: Tesseract not found."
    except Exception as e:
        return f"Error during OCR: {e}"

@app.route('/', methods=['GET'])
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/ocr', methods=['POST'])
def ocr_route():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided.'}), 400

        image_data = data['image']
        preprocessed_image = preprocess_image(image_data)

        if preprocessed_image is not None:
            extracted_text = extract_text(preprocessed_image)
            return jsonify({'text': extracted_text})
        else:
            return jsonify({'error': 'Image preprocessing failed.'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.json
    image_data = data.get('image_data')
    if not image_data:
        return jsonify({"error": "No image data provided"}), 400

    processed_image = preprocess_image(image_data)
    if processed_image is None:
        return jsonify({"error": "Failed to process image"}), 500

    # Perform OCR on the processed image
    text = pytesseract.image_to_string(processed_image)
    return jsonify({"text": text})

if __name__ == '__main__':
    app.run(debug=True)