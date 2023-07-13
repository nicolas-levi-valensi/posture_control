from __future__ import annotations
import signal
import numpy as np
import cv2
from nico_lib.hvc_minilib import HandVideoClassifier

# SETTINGS
MODEL_PATH = "Assets/model_data/model.h5"  # TensorFlow Keras model path root
MODEL_OUTPUT_LABELS = ["dummy", "down", "left", "dummy", "dummy", "pinch", "right", "dummy", "dummy", "dummy", "up"]
KEY_PRESS_DELAY = 0.2  # Delay between each press on [up, down, left, right] key
USE_VERBOSE_ON_HVC = True  # Enables INFO output from HandVideoClassifier

MAX_DIST_TO_GRAB = 20
GRAB_INDEX = 0


class Ball:
    def __init__(self,
                 position: list | np.ndarray,
                 ball_radius: int = 30,
                 color: tuple | list | np.ndarray = (255, 255, 255)):
        self.position = position
        self.ball_radius = ball_radius
        self.color = color
        self.grabbed = False
        self.grabbed_by = 0

    def draw(self, src):
        cv2.circle(src,
                   center=self.position,
                   radius=self.ball_radius,
                   color=self.color,
                   thickness=-1,
                   lineType=cv2.LINE_4)

    def set_position(self, position: list | np.ndarray):
        self.position = position

    def get_position(self):
        return self.position

    def set_grabbed(self, grabbed: bool, holder_index: int = 0):
        self.grabbed = grabbed
        self.grabbed_by = holder_index

    def is_grabbed(self):
        return self.grabbed


class Box:
    def __init__(self,
                 position: list | np.ndarray,
                 box_size: tuple | list | np.ndarray = (100, 40),
                 color: tuple | list | np.ndarray = (255, 255, 255)):
        self.position = position
        self.box_size = box_size
        self.color = color
        self.grabbed = False
        self.grabbed_by = 0

    def draw(self, src):
        cv2.rectangle(src,
                      pt1=(self.position[0] - self.box_size[0] // 2, self.position[1] - self.box_size[1] // 2),
                      pt2=(self.position[0] + self.box_size[0] // 2, self.position[1] + self.box_size[1] // 2),
                      color=self.color,
                      thickness=-1,
                      lineType=cv2.LINE_4)

    def set_position(self, position: list | np.ndarray):
        self.position = position

    def get_position(self):
        return self.position

    def set_grabbed(self, grabbed: bool, holder_index: int = 0):
        self.grabbed = grabbed
        self.grabbed_by = holder_index

    def is_grabbed(self):
        return self.grabbed


def draw_hands(src, coords):
    cv2.circle(src,
               center=coords[0],
               radius=10,
               color=(255, 0, 0),
               thickness=3,
               lineType=cv2.LINE_4)

    cv2.circle(src,
               center=coords[1],
               radius=10,
               color=(0, 0, 255),
               thickness=3,
               lineType=cv2.LINE_4)


def main():
    hvc = HandVideoClassifier(model_path=MODEL_PATH, stream_path=0, video_output=False,
                              labels_on_vid=MODEL_OUTPUT_LABELS, verbose=USE_VERBOSE_ON_HVC).start()

    signal.signal(signal.SIGINT, hvc.stop)

    objects = []

    box_0 = Box(position=[200, 200],
                box_size=[40, 30],
                color=(100, 200, 0))
    objects.append(box_0)
    box_1 = Box(position=[300, 100],
                box_size=[80, 60],
                color=(0, 100, 250))
    objects.append(box_1)

    ball_0 = Ball(position=[50, 400],
                  ball_radius=30,
                  color=(30, 30, 10))
    objects.append(ball_0)

    prev_states = -1
    while hvc.is_running():
        states = hvc.get_prediction()
        hands_coords = hvc.get__hands_coords()
        hmi_output = np.zeros((480, 640, 3))

        for obj in objects:
            held = False
            for hand_pos, hand_id in zip(hands_coords, range(2)):
                if (np.abs(np.subtract(obj.get_position(), hand_pos)) < MAX_DIST_TO_GRAB).all() \
                        and states[hand_id] == GRAB_INDEX and prev_states[hand_id] != GRAB_INDEX:
                    obj.set_grabbed(grabbed=True, holder_index=hand_id)
                    obj.set_position(position=hand_pos)
                    held = True
                elif obj.is_grabbed() and hand_id == obj.grabbed_by and states[hand_id] == GRAB_INDEX:
                    obj.set_position(position=hand_pos)
                    held = True
            if not held:
                obj.set_grabbed(grabbed=False)

        for obj in objects:
            obj.draw(hmi_output)

        draw_hands(hmi_output, hands_coords)

        prev_states = states

        cv2.imshow("HMI", hmi_output)
        if cv2.waitKey(1) == 27:
            hvc.stop()
            break

    cv2.destroyWindow("HMI")


if __name__ == '__main__':
    main()
