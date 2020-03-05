#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Authors: klin
Email: l33klin@foxmail.com
Date: 2020/3/3
"""

import base64
import os
import re
from sanic import Sanic
from sanic.log import logger
from sanic.response import json
import tensorflow as tf
import numpy as np

# IMG_PATH = "/Users/leek/Downloads/SCENERY"
IMG_PATH = "/Users/klin/Library/Mobile Documents/com~apple~Preview/Documents/Scenery"
# IMG_PATH = "."
IMG_SUFFIX = ".jpeg"

app = Sanic()

# model settings
model = None
IMG_HEIGHT = IMG_WIDTH = 150

_headers = {"Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Headers": "Content-Type, Access-Control-Allow-Headers, X-Requested-With"}


def confirm_name(name):
    exists_images = os.listdir(IMG_PATH)
    max_num = -1
    for ef in exists_images:
        if ef.endswith(IMG_SUFFIX):
            if re.match("{}\d?".format(name), ef):
                num_str = ef.split(IMG_SUFFIX)[0].split(name)[1]
                num = int(num_str) if num_str else 0
                max_num = num if max_num < num else max_num
    return "".join([name, str(max_num+1)]) if max_num != -1 else name


def save_img(img_str, name):
    if img_str.startswith("data:image/jpeg"):
        img_str = img_str.split("data:image/jpeg;base64,")[1]
    imgdata = base64.decodebytes(img_str.encode())
    img_path = os.path.join(IMG_PATH, "{}{}".format(confirm_name(name), IMG_SUFFIX))
    logger.info("save image to: {}".format(img_path))
    with open(img_path, "wb") as f:
        f.write(imgdata)
    return img_path


def load_model(model_path):
    global model
    # Recreate the exact same model
    model = tf.keras.models.load_model(model_path)
    print("Load model success...")


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image /= 255.0  # normalize to [0,1] range
    image = np.expand_dims(image, axis=0)
    return image


def predict(bs64_img):
    imgdata = base64.decodebytes(bs64_img.encode('utf8'))
    image = preprocess_image(imgdata)
    predictions = model.predict([image])
    return predictions[0][0]


@app.route('/')
async def test(request):
    return json({'hello': 'world'})


@app.post('/predict')
async def test(request):
    img_str = request.json.get('bs64_img')
    logger.info("image size: {0:.1f} KB".format(len(img_str.encode()) / 1024))
    prediction = float(predict(img_str))
    print("prediction: ", prediction)
    return json({"prediction": prediction})


@app.options('/predict')
async def test(request):
    return json({'hello': 'world'},
                headers=_headers,
                status=200)

if __name__ == '__main__':
    load_model('./model/1')
    app.run(host='0.0.0.0', port=8007)
