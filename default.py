import os
from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Define the route for prediction
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        # Save the file to the uploads folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        # Open the file using PIL
        img = Image.open(file_path)
        # Preprocess the image
        img = img.resize((224, 224))
        x = np.asarray(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # Make a prediction
        preds = model.predict(x)
        preds = decode_predictions(preds, top=1)[0]
        prediction = preds[0][1]
        # Render the result page
        return render_template('result.html', prediction=prediction)
    # Render the upload page
    return render_template('upload.html')
