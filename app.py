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

def preprocess_image(image_data, column_option):
    """Loads and preprocesses the image for better OCR results."""
    try:
        # Decode base64 image data
        decoded_data = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(decoded_data))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        save_image(gray, "preprocessed", "grayscale")

        if column_option == '2':
            # Split the image into two columns with a margin
            height, width = gray.shape
            mid = width // 2
            margin = 10  # Adjust the margin size as needed
            left_column = gray[:, :mid - margin]
            right_column = gray[:, mid + margin:]
            save_image(left_column, "preprocessed", "left_column")
            save_image(right_column, "preprocessed", "right_column")
            return left_column, right_column
        else:
            return gray, None
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        return None, None

def extract_text(image):
    try:
        pytesseract.pytesseract.tesseract_cmd = 'C:\\Tesseract-OCR\\tesseract.exe'  # Or the full path on Windows
        custom_config = r'--oem 3 --psm 1 -l mar+eng' # Adjust PSM if needed
        text = pytesseract.image_to_string(image, config=custom_config)
        text = text.replace('\n', ' ')  # Replace newlines with spaces
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
        column_option = data.get('column_option', '1')
        preprocessed_image_left, preprocessed_image_right = preprocess_image(image_data, column_option)

        if preprocessed_image_left is not None:
            extracted_text_left = extract_text(preprocessed_image_left)
            extracted_text_right = extract_text(preprocessed_image_right) if preprocessed_image_right is not None else ""
            extracted_text = extracted_text_left + " " + extracted_text_right
            return jsonify({'text': extracted_text})
        else:
            return jsonify({'error': 'Image preprocessing failed.'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.json
    image_data = data.get('image_data')
    column_option = data.get('column_option', '1')
    if not image_data:
        return jsonify({"error": "No image data provided"}), 400

    processed_image_left, processed_image_right = preprocess_image(image_data, column_option)
    if processed_image_left is None:
        return jsonify({"error": "Failed to process image"}), 500

    # Perform OCR on the processed image
    text_left = pytesseract.image_to_string(processed_image_left)
    text_right = pytesseract.image_to_string(processed_image_right) if processed_image_right is not None else ""
    text = text_left + " " + text_right
    return jsonify({"text": text})

if __name__ == '__main__':
    app.run(debug=True)