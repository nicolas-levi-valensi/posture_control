import time
import signal
from pynput.keyboard import Controller, Key
from nico_lib.hvc_minilib import HandVideoClassifier

# SETTINGS
MODEL_PATH = "Assets/model_data/model.h5"  # TensorFlow Keras model path root
MODEL_OUTPUT_LABELS = ["dummy", "down", "left", "dummy", "Enter", "dummy", "right", "dummy", "dummy", "up"]
KEY_PRESS_DELAY = 0.2  # Delay between each press on [up, down, left, right] key
USE_VERBOSE_ON_HVC = True  # Enables INFO output from HandVideoClassifier


def directional_arrow_control(state, controller):
    if state != -1:
        if state == 1:  # down
            controller.press(Key.down)
            controller.release(Key.down)
        elif state == 6:  # right
            controller.press(Key.right)
            controller.release(Key.right)
        elif state == 2:  # left
            controller.press(Key.left)
            controller.release(Key.left)
        elif state == 9:  # up
            controller.press(Key.up)
            controller.release(Key.up)
        elif state == 4:  # Enter
            controller.press(Key.enter)
            controller.release(Key.enter)
            time.sleep(1)  # delay to avoid multiple presses on key
        time.sleep(KEY_PRESS_DELAY)
    else:
        pass


def main():
    hvc = HandVideoClassifier(model_path=MODEL_PATH, stream_path=0, video_output=True,
                              labels_on_vid=MODEL_OUTPUT_LABELS, verbose=USE_VERBOSE_ON_HVC).start()

    signal.signal(signal.SIGINT, hvc.stop)

    controller = Controller()
    while hvc.is_running():
        state = hvc.get_prediction()
        directional_arrow_control(state, controller)


if __name__ == '__main__':
    main()
