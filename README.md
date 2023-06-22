# Drone Control

This package contains all requirements to add new classes, train those classes and use them as control with pynput is respectively 3 python scripts

* Dataset class registering : **create_dataset.py**
* Model training and visualizer :  **train.py**
  * train.py **--no_train** argument to avoid training and use visualizer only
* Controller : **Controller.py**.

Default configuration :
* *Up, down, left, right* hand posture with index finger for directional arrows.
* *Thumb and index touching* to press enter

Existing classes are present in [Assets/datasets_records/](Assets/datasets_records).

\*.csv files are not necessary for the controller as the model is pretrained 
but the labels corresponding to the model output have to be corresponding.
The model output is trained with the \*.csv files sorted alphabetically.
