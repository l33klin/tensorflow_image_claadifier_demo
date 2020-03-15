#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Authors: klin
Email: l33klin@foxmail.com
Date: 2020/3/3
"""
import os
import time
import random
import getpass

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


USER = getpass.getuser()

MODEL_PATH = "./model/2"  # the path which you save your model after train

BITE_IMG = "bite.jpg"
NO_BITE_IMG = "no_bite.jpg"

IMG_HEIGHT = IMG_WIDTH = 150


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image /= 255.0  # normalize to [0,1] range
    image = np.expand_dims(image, axis=0)
    
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


# Recreate the exact same model
new_model = tf.keras.models.load_model(MODEL_PATH)

# Check that the state is preserved
# new_predictions = new_model.predict([load_and_preprocess_image(BITE_IMG)])

# DATASET_DIR = "/Users/{}/Nextcloud/Documents/MineCraft/2020-02-26/dataset".format(USER)
DATASET_DIR = "./dataset"

BITE_DIR = os.path.join(DATASET_DIR, "validation/bite")
NO_BITE_DIR = os.path.join(DATASET_DIR, "validation/no_bite")

BITE_IMG_5 = random.sample(os.listdir(BITE_DIR), 5)
NO_BITE_IMG_5 = random.sample(os.listdir(NO_BITE_DIR), 5)

# BITE_IMG_LIST = [load_and_preprocess_image(os.path.join(BITE_DIR, img)) for img in BITE_IMG_5]
# NO_BITE_IMG_LIST = [load_and_preprocess_image(os.path.join(NO_BITE_DIR, img)) for img in NO_BITE_IMG_5]
start = time.time()

for img in BITE_IMG_5:
    predictions = new_model.predict([load_and_preprocess_image(os.path.join(BITE_DIR, img))])
    print("predictions: {}".format(predictions))

for img in NO_BITE_IMG_5:
    predictions = new_model.predict([load_and_preprocess_image(os.path.join(NO_BITE_DIR, img))])
    print("predictions: {}".format(predictions))

print("Cost time: {}".format(time.time() - start))


def get_file_name(file_path):
    name_suf = os.path.split(file_path)[-1]
    if "." in name_suf:
        return ".".join(name_suf.split('.')[:-1])
    else:
        return name_suf


start = time.time()

IMG_ALL = [os.path.join(BITE_DIR, x) for x in os.listdir(BITE_DIR)] \
          + [os.path.join(NO_BITE_DIR, x) for x in os.listdir(NO_BITE_DIR)]

IMG_ALL = sorted(IMG_ALL, key=get_file_name)
xValue = list(range(0, len(IMG_ALL)))
yValue = []
for img in IMG_ALL:
    predictions = new_model.predict([load_and_preprocess_image(img)])
    yValue.append(predictions[0][0])

print("predict all cost time: {}".format(time.time() - start))

plt.title('Bite scatter')
plt.legend()
plt.scatter(xValue, yValue, s=20, c="#ff1212", marker='o')
# plt.scatter(no_bite_xValue, no_bite_yValue, s=20, c="b", marker='o')
plt.show()
