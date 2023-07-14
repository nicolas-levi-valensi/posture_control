from __future__ import annotations
import signal
import numpy as np
import cv2
from nico_lib.hvc_minilib import HandVideoClassifier
from nico_lib.hmi_minilib import Ball, Box, HMI

# SETTINGS
MODEL_PATH = "Assets/model_data/model.h5"  # TensorFlow Keras model path root
MODEL_OUTPUT_LABELS = ["dummy", "down", "left", "dummy", "dummy", "pinch", "right", "dummy", "dummy", "dummy", "up"]
KEY_PRESS_DELAY = 0.2  # Delay between each press on [up, down, left, right] key
USE_VERBOSE_ON_HVC = True  # Enables INFO output from HandVideoClassifier

MAX_DIST_TO_GRAB = 20
GRAB_INDEX = 0
ADD_BALL_INDEX = 10
ADD_BOX_INDEX = 9
DEL_INDEX = 5


def main():
    hvc = HandVideoClassifier(model_path=MODEL_PATH, stream_path=0, video_output=False,
                              labels_on_vid=MODEL_OUTPUT_LABELS, verbose=USE_VERBOSE_ON_HVC).start()

    hmi = HMI(window_name="Interface")

    signal.signal(signal.SIGINT, hvc.stop)

    box_0 = Box(position=[200, 200],
                box_size=[40, 30],
                color=(100, 200, 0))
    hmi.add_object(box_0)

    box_1 = Box(position=[300, 100],
                box_size=[80, 60],
                color=(0, 100, 250))
    hmi.add_object(box_1)

    ball_0 = Ball(position=[50, 400],
                  ball_radius=30,
                  color=(255, 0, 100))
    hmi.add_object(ball_0)

    prev_states = [-1, -1]
    while hvc.is_running():
        states = hvc.get_prediction()
        hands_coords = hvc.get__hands_coords()
        hmi.set_hands_coords(hands_coords)

        # Grab test
        for hand_pos, hand_id in zip(hands_coords, range(2)):
            for obj in hmi.objects:
                held = False
                if (np.less(np.abs(np.subtract(obj.get_position(), hand_pos)), obj.get_hit_box())).all() \
                        and states[hand_id] == GRAB_INDEX and prev_states[hand_id] != GRAB_INDEX:
                    obj.set_grabbed(grabbed=True, holder_index=hand_id)
                    obj.set_position(position=hand_pos)
                    held = True
                elif obj.is_grabbed() and hand_id == obj.grabbed_by and states[hand_id] == GRAB_INDEX:
                    obj.set_position(position=hand_pos)
                    held = True
                if not held and hand_id == obj.grabbed_by:
                    obj.set_grabbed(grabbed=False)
                elif held:
                    break

        for hand_pos, hand_id in zip(hands_coords, range(2)):
            for obj, obj_id in zip(hmi.objects, range(len(hmi.objects))):
                if (np.less(np.abs(np.subtract(obj.get_position(), hand_pos)), obj.get_hit_box())).all() \
                        and states[hand_id] == DEL_INDEX and prev_states[hand_id] != DEL_INDEX:
                    hmi.delete_object(obj_id)
                    break

        # Adding balls on hand position if ADD_BALL_INDEX state is reached by hand
        for state, prev_state, xy in zip(states, prev_states, hands_coords):
            if state == ADD_BALL_INDEX and prev_state != ADD_BALL_INDEX:
                hmi.add_object(Ball(position=xy, ball_radius=20, color=np.random.random(size=3)))

        # Adding boxes on hand position if ADD_BOX_INDEX state is reached by hand
        for state, prev_state, xy in zip(states, prev_states, hands_coords):
            if state == ADD_BOX_INDEX and prev_state != ADD_BOX_INDEX:
                hmi.add_object(Box(position=xy, box_size=[40, 50], color=np.random.random(size=3)))

        prev_states = states

        hmi.draw()
        if cv2.waitKey(1) == 27:
            hvc.stop()
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
