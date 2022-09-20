#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kiran
"""
import os
from flask import Flask, render_template, request #, url_for
from time import sleep

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
import numpy as np

# print(os.getcwd())
# from pathlib import Path
# print(Path.cwd())
# print(os.path.dirname(os.path.abspath(__file__)))

UPLOAD_FOLDER = '/home/kiran/Desktop/My Computer/Work/AI/Work files/gender-age-prediction-live/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, 
            template_folder='/home/kiran/Desktop/My Computer/Work/AI/Work files/gender-age-prediction-live/templates',
            #static_url_path='',
            static_folder="/home/kiran/Desktop/My Computer/Work/AI/Work files/gender-age-prediction-live/static"
            )
app.templates_auto_reload = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
def extract_features(image):
    features = []
    img = load_img(image, grayscale=True)
    img = img.resize((128, 128), Image.ANTIALIAS)
    img = np.array(img)
    features.append(img)
    
    features = np.array(features)
    features = features.reshape(len(features), 128, 128, 1)

    return features
           
@app.route('/')
def index():
    static_image_1 = app.config['UPLOAD_FOLDER']+'Me-kiran.jpeg'
    return render_template("index.html", static_image_1=static_image_1)

# @app.route('/submit')
# def submit():
#     return render_template("submit.html")

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':

        file = request.files['image']
        
        if file and allowed_file(file.filename):
            #filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'image.jpeg'))
            sleep(1)
            
            gender_dict = {0:'Male', 1:'Female'}
            model = tf.keras.models.load_model(app.config['UPLOAD_FOLDER']+'human_gender_age_prediction.h5')
            
            X1 = extract_features(app.config['UPLOAD_FOLDER']+'image.jpeg')
            X1 = X1 / 255.0
            
            pred = model.predict(X1.reshape(1, 128, 128, 1))
            pred_gender = gender_dict[round(pred[0][0][0])]
            pred_age = round(pred[1][0][0])
            # prediction = f"Predicted Gender: {pred_gender}, Predicted Age: {pred_age}"
            
            return render_template('submit.html', gender=pred_gender, age=pred_age)
        
       
if __name__ == "__main__":
    app.run()




