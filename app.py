from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import io
import numpy as np
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the Keras model
model = load_model('best.h5')
logging.debug("Model loaded successfully.")

def preprocess_image(image):
    """Preprocesses an image for the model.
    
    Args:
        image: A PIL Image object.
        
    Returns:
        A NumPy array representing the preprocessed image with the expected shape
        for the model.
    """
    logging.debug("Starting image preprocessing.")
    
    # Get the expected input shape from the model
    input_shape = model.input_shape  # Includes batch dimension
    logging.debug(f"Model input shape: {input_shape}")

    if len(input_shape) == 4:  # Assuming (batch_size, height, width, channels)
        _, height, width, channels = input_shape
    else:
        raise ValueError(f"Unexpected input shape format: {input_shape}")

    # Resize the image to match the model's expected input size
    image = image.resize((width, height))  # Width, height
    logging.debug("Image resized successfully.")

    # Convert to a NumPy array
    image = np.array(image)
    logging.debug(f"Image array shape after conversion: {image.shape}")

    # Handle different number of channels (assuming grayscale or RGB)
    if len(image.shape) == 2:  # Grayscale
        # Convert to RGB by duplicating each channel
        image = np.expand_dims(image, axis=-1)
        image = np.repeat(image, 3, axis=-1)
        logging.debug("Grayscale image converted to RGB.")
    elif len(image.shape) == 3 and image.shape[2] != channels:
        raise ValueError(f"Unexpected number of channels: {image.shape[2]}. Expected {channels}")

    # Normalize pixel values (adjust normalization range if needed)
    image = image / 255.0
    logging.debug("Image normalized successfully.")

    # Add a batch dimension
    image = np.expand_dims(image, axis=0)
    logging.debug("Batch dimension added successfully.")

    return image

@app.route('/')
def hello_world():
    logging.debug("Serving the main page.")
    return render_template("forest_fire.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            logging.debug("No image in request files.")
            return render_template('forest_fire.html', pred='No image uploaded')

        file = request.files['image']
        if file.filename == '':
            logging.debug("No image selected.")
            return render_template('forest_fire.html', pred='No image selected')

        image = Image.open(io.BytesIO(file.read()))
        logging.debug("Image opened successfully.")

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Predict using the loaded model
        prediction = model.predict(processed_image)[0]
        logging.debug(f"Model prediction: {prediction}")
        
        # Since the model returns a single output
        output = prediction[0]

        if output >= 0.5:
            return render_template('forest_fire.html', pred=f'Pneumonia detected.\nProbability of pneumonia is {output:.2f}', bhai="Take necessary precautions!")
        else:
            return render_template('forest_fire.html', pred=f'Normal.\nProbability of pneumonia is {output:.2f}', bhai="No signs of pneumonia detected.")
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return render_template('forest_fire.html', pred=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
