# Keyboard Control

## Description
This package contains all requirements to add new classes, 
train those classes and use them as control with pynput in respectively 3 python scripts

* **create_dataset.py** : dataset class registering.
* **train.py** : Model training and visualizer.
  * `--no_train` argument to avoid training and use visualizer only.
* **Controller.py** : Example controller using the pre-trained model and Mediapipe detection.

### Default configuration
*Up, down, left, right* hand posture with index finger for directional arrows.
*Thumb and index touching* to press enter

Existing classes are present in [Assets/datasets_records/](Assets/datasets_records).

*.csv files are not necessary for the controller as the model is pretrained 
but the labels corresponding to the model output have to be corresponding.
The model output is trained with the *.csv files sorted alphabetically.

### Used Packages

* **Mediapipe** - hand landmarks acquisition.
* **OpenCV** - low-level image processing and display.
* **TensorFlow - Keras** - Direct Neural Network for posture prediction.
* **Pynput** - Keyboard simulation based on predictions.
* **Tkinter** - User interface.