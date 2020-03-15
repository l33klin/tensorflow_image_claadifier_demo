# Tensorflow Image Classifier Demo
A image classifier base on tensorflow: train, predict and serve demo.

## Prepare
Before the train, you should prepare your dataset, and organized them as `./dataset` path:
```bash
dataset
├── train
│   ├── bite
│   │   ├── 2020-02-26_17-27_frame_2232.jpg
│   │   ├── 2020-02-26_17-27_frame_2232_shift_1.jpg
│   │   ...
│   │
│   └── no_bite
│       ├── 2020-02-26_17-27_frame_2202.jpg
│       ├── 2020-02-26_17-27_frame_2202_shift_1.jpg
│       ...
│
└── validation
    ├── bite
    │   ├── 2020-02-26_17-27_frame_2226.jpg
    │   ├── 2020-02-26_17-27_frame_2226_shift_1.jpg
    │   ...
    │
    └── no_bite
        ├── 2020-02-26_17-27_frame_2184.jpg
        ├── 2020-02-26_17-27_frame_2184_shift_1.jpg
        ...
```

## Train and Save model
Then train and save your model
```bash
$ python train.py

```

## Predict

## Deploy as server

## Drawbacks
1. only support two kinds of image classify
2. method of save_model will deprecated in later version
3. no GPU support
...

## Reference
1. [Image classification | TensorFlow Core](https://www.tensorflow.org/tutorials/images/classification)
2. [tf.keras.models.save_model | TensorFlow Core v2.1.0](https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model)
3. [Sanic — Sanic 19.12.2 documentation](https://sanic.readthedocs.io/en/latest/)
