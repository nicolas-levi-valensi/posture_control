# Hand posture control

## Description
This package contains all requirements to add new classes, 
train those classes and use them as control with pynput in respectively 3 python scripts

1. **create_dataset.py** : dataset class registering.
2. **train.py** : Model training and visualizer.
   1. `python3 train.py` without arguments to train/re-train the model before visualizing.
   2. `python3 train.py --no_train` argument to avoid training and use visualizer only.
3. **Controller.py** : Example controller using the pre-trained model and Mediapipe detection.

#### Demonstration video

[![Demonstration video](https://i3.ytimg.com/vi/3sla-qnNxwM/maxresdefault.jpg)](https://www.youtube.com/watch?v=3sla-qnNxwM)

### Default configuration of the controller
*Up, down, left, right* hand posture with index finger for directional arrows.
*Thumb and index touching* to press enter

Existing posture classes are present in [Assets/datasets_records](Assets/datasets_records).

The *.csv files are not necessary for the controller as the model is pretrained 
but the labels have to correspond to the model output.
The model output is trained with the *.csv files sorted alphabetically.

Escape key while focused on OpenCV video output window to end the process by default (key code 27).

### Used Packages

* **Mediapipe** - hand landmarks acquisition.
* **OpenCV** - low-level image processing and display.
* **TensorFlow - Keras** - Direct Neural Network for posture prediction.
* **Pynput** - Keyboard simulation based on predictions.
* **Tkinter** - User interface.

## HandVideoClassifier Class

### Class usage

The [HandVideoClassifier (HVC)](nico_lib/hvc_minilib.py) class handles :

1. **Video capture** from camera or source file 
*(will be updated with video buffering instead of direct output to avoid time differed prediction)*.
2. **Real time prediction** based on trained model.
3. **Video output** with optional labels on frame for debugging.
4. **Verbosity control** (Subprocess state information)

The labels shown on video can be passed in the `labels_on_vid` optional argument.

### Methods and attributes

#### Initialisation

```python
from nico_lib.hvc_minilib import HandVideoClassifier

hvc = HandVideoClassifier(model_path="Assets/model_data",
                          stream_path=0,  # To use camera at port 0 (default)
                          video_output=True,  # or a list/1D-array of labels
                          verbose=True,  # outputs subprocess behavior to console
                          labels_on_vid=None,  # Or a list/1D-array of labels
                          always_on_top=True)  # keeps video output in front of other apps
```

#### Process startup

```python
from nico_lib.hvc_minilib import HandVideoClassifier
hvc = HandVideoClassifier("Assets/model_data")

hvc.start()  # Begins acquisition subprocess
```

#### Get prediction

```python
from nico_lib.hvc_minilib import HandVideoClassifier
hvc = HandVideoClassifier("Assets/model_data").start()

prediction = hvc.get_prediction()  # Returns the prediction made on last frame
```

#### Get running state

```python
from nico_lib.hvc_minilib import HandVideoClassifier
hvc = HandVideoClassifier("Assets/model_data").start()

state = hvc.is_running()  # Returns the running state of the detection subprocess (bool)
```

#### Stop subprocess and release video source

```python
from nico_lib.hvc_minilib import HandVideoClassifier
hvc = HandVideoClassifier("Assets/model_data").start()

hvc.stop()
```

#### Attributes

```python
from nico_lib.hvc_minilib import HandVideoClassifier
hvc = HandVideoClassifier("Assets/model_data").start()

labels = hvc.labels  # Retrieve the labels passed in argument
model_path = hvc.model_path  # Retrieve the model path passed in argument
```
