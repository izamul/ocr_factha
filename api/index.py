import cv2
import numpy as np
from flask import Flask
from keras.models import load_model

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/about')
def about():
    return 'About'

if __name__ == "__main__":
    app.run(debug=True)