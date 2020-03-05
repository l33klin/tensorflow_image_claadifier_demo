#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Authors: klin
Email: l33klin@foxmail.com
Date: 2020/3/3
"""
import requests
import base64
import time


img_path = 'bite.jpg'
_url = 'http://127.0.0.1:8007/predict'


start = time.time()
encoded_string = None
with open(img_path, 'rb') as f:
    encoded_string = base64.b64encode(f.read()).decode('utf8')

body = {
    "bs64_img": encoded_string
}

# print(body)

response = requests.post(_url, json=body)
print("cost time: {}".format(time.time() - start))
print(response.content)
