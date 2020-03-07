#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Authors:    klin
Email:      l33klin@foxmail.com
Date:       2020/2/26
Ref:        https://www.tensorflow.org/tutorials/images/classification
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import getpass
import numpy as np
import matplotlib.pyplot as plt


batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150


# _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
#
# path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
#
# PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

PATH = "/Users/{}/Nextcloud/Documents/MineCraft/2020-02-26/dataset2".format(getpass.getuser())

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_bite_dir = os.path.join(train_dir, 'bite')  # directory with our training cat pictures
train_no_bite_dir = os.path.join(train_dir, 'no_bite')  # directory with our training dog pictures
validation_bite_dir = os.path.join(validation_dir, 'bite')  # directory with our validation cat pictures
validation_no_bite_dir = os.path.join(validation_dir, 'no_bite')  # directory with our validation dog pictures

num_bite_tr = len(os.listdir(train_bite_dir))
num_no_bite_tr = len(os.listdir(train_no_bite_dir))

num_bite_val = len(os.listdir(validation_bite_dir))
num_no_bite_val = len(os.listdir(validation_no_bite_dir))

total_train = num_bite_tr + num_no_bite_tr
total_val = num_bite_val + num_no_bite_val

print('total training bite images:', num_bite_tr)
print('total training no_bite images:', num_no_bite_tr)

print('total validation bite images:', num_bite_val)
print('total validation no_bite images:', num_no_bite_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

sample_training_images, _ = next(train_data_gen)


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# plotImages(sample_training_images[:5])

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, name="predictions")
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()
input("Press enter to train...")

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# Save model
MODEL_DIR = "./model"
version = 2
export_path = os.path.join(MODEL_DIR, str(version))

# REF: https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model
tf.keras.models.save_model(
    model,
    export_path,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)

print('\nSaved model: {}'.format(export_path))


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)
