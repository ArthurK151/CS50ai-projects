# Traffic Sign Recognition Experimentation

## Overview
In this project, I implemented `load_data` and `get_model` in `traffic.py` using OpenCV (`cv2`) and TensorFlow (`tf.keras`) to classify images from the [German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/?section=gtsrb&subsection=news) (GTSRB) across 43 road sign categories. In `load_data`, each image was read as a NumPy array, resized to `(IMG_WIDTH, IMG_HEIGHT)` (`30x30` pixels by default), and paired with its integer category label.

## Experimentation Process

### Baseline Model (Two Conv/Pool Blocks + Dense + Dropout)
I started with a convolutional neural network containing two `Conv2D` layers (32 and 64 filters with `3x3` kernels and `ReLU` activation), each followed by a `2x2` `MaxPooling2D` layer, a `Flatten` layer, a `128`-unit `Dense` hidden layer with `ReLU` activation, `Dropout(0.5)`, and an output layer of `NUM_CATEGORIES` (43) units with `softmax` activation, compiled with `adam` and `categorical_crossentropy`. This baseline trained steadily across epochs and reached strong test accuracy (around **95%**).

### What Worked Well vs. What Didn't Work Well
* **What worked well:** Pairing `MaxPooling2D` with `Dropout(0.5)` before the output layer helped curb overfitting on the training set while keeping generalization high on the test split. Resizing all images consistently to `(30, 30, 3)` allowed batch training without shape mismatches.
* **What I noticed / challenges:** Without dropout or with too many dense units, training accuracy climbed quickly while test accuracy plateaued earlier, showing signs of overfitting. Deeper networks or higher filter counts also noticeably increased training time per epoch on CPU, especially when loading and processing the full 43-category dataset. Smaller `3x3` convolutional kernels captured fine sign details (such as speed limit digits and borders) effectively while keeping parameter counts manageable.
