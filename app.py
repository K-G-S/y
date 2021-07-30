from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import time
from flask import *
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import requests
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, MaxPool2D, BatchNormalization, Flatten
from PIL import Image
import cv2
import json

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(30,30,3)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.5))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.load_weights('static/model.h5')

COUNT = 0
app = Flask(__name__)
g_model = None
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

# Classes of meters
classes = { 0:'Real',
            1:'Spoof', 
            }
def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

def image_processing(img):
    global g_model
    if g_model is None:
        g_model = load_model('Knight.h5')
        print("model init")
    data=[]
    image = Image.open(img)
    image = image.resize((100,100))
    data.append(np.array(image))
    X_test=np.array(data)
    Y_pred = g_model.predict_classes(X_test)
    print(Y_pred)
    return Y_pred

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        # Make prediction
        result = image_processing(file_path)
        s = [str(i) for i in result]
        a = int("".join(s))
        result = classes[a]
        os.remove(file_path)
        return  {"real": float(1-a), "spoof": float(a), "nonmeter": 0}

    elif request.method == 'GET' and request.args.get('url', default=None):
        url = request.args.get('url', default=None)
        r = requests.get(url, allow_redirects=True)
        filename = str(time.time()) + '.jpg'
        print(filename)
        open(filename, 'wb').write(r.content)
        # Make prediction
        result = image_processing(file_path)
        s = [str(i) for i in result]
        a = int("".join(s))
        result = classes[a]
        os.remove(file_path)
        return result
    return None



@app.route('/home', methods=['GET'])
def home():
    return render_template('main.html')


@app.route('/testimage', methods=['POST'])
def testimage():
    
    if request.method == 'POST':
        global COUNT
        img = request.files['file']
        img.save('Saved Test Images/{}.jpg'.format(COUNT))    
        img_arr = cv2.imread('Saved Test Images/{}.jpg'.format(COUNT))

        img_arr = cv2.resize(img_arr, (30,30))
        img_arr = img_arr / 255.0
        img_arr = img_arr.reshape(1, 30,30,3)
        prediction = model.predict(img_arr)

        x = round(prediction[0,0], 2)
        y = round(prediction[0,1], 2)
        z = round(prediction[0,2], 2)
        preds = np.array([x,y,z])
        COUNT += 1
        return {"real": float(preds[0]), "spoof": float(preds[1]), "nonmeter": float(preds[2])}

#        if preds[1] > 0.50:
 #           return 'Meter is Spoof'
  #      elif preds[2]  > 0.50:
   #         return 'METER IMAGE IS NON METER'
    #    else:
     #       return 'Meter Image is Real'
    #return None

@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('Saved Test Images', "{}.jpg".format(COUNT-1))

if __name__ == '__main__':
    g_model = load_model('static/Knight.h5')
    app.run(host='0.0.0.0',port=80,debug=True)
