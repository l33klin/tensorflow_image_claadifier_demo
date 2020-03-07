#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Authors: klin
Email: l33klin@foxmail.com
Date: 2020/3/3
"""

import base64
import time
from sanic import Sanic
from sanic.log import logger
from sanic.response import json
import tensorflow as tf
import numpy as np

app = Sanic()

# model settings
model = None
IMG_HEIGHT = IMG_WIDTH = 150


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
async def default_page(request):
    return json({'hello': 'world'})


@app.post('/predict')
async def foo(request):
    img_str = request.json.get('bs64_img')
    # logger.info("image size: {0:.1f} KB".format(len(img_str.encode()) / 1024))
    start = time.time()
    prediction = float(predict(img_str))
    logger.info("prediction: {0:.4f}, cost: {1:.4f}".format(prediction, time.time()-start))
    return json({"prediction": prediction})


if __name__ == '__main__':
    load_model('./model/2')
    app.run(host='0.0.0.0', port=8007)
