import time
import signal
from nico_lib.hvc_minilib import HandVideoClassifier, DroneHandVideoClassifier
from djitellopy import Tello

# SETTINGS
MODEL_PATH = "Assets/model_data/"  # TensorFlow Keras model path root
MODEL_OUTPUT_LABELS = ["dummy", "down", "left", "dummy", "Enter", "dummy", "right", "dummy", "dummy", "dummy", "up"]
PREDICTION_DELAY = 0.2  # Delay between each press on [up, down, left, right] key
USE_VERBOSE_ON_HVC = True  # Enables INFO output from HandVideoClassifier


def drone_commands(drone: Tello, state: int):
    if state != -1:
        if state == 1:  # down
            drone.move_down(x=50)
            time.sleep(2)
        elif state == 6:  # right
            drone.move_right(x=50)
            time.sleep(2)
        elif state == 2:  # left
            drone.move_left(x=50)
            time.sleep(2)
        elif state == 10:  # up
            drone.move_up(x=50)
            time.sleep(2)
        elif state == 4:  # Enter
            drone.rotate_counter_clockwise(x=360)
            time.sleep(5)  # delay to avoid multiple presses on key
        time.sleep(PREDICTION_DELAY)


def webcam_main():
    hvc = HandVideoClassifier(model_path=MODEL_PATH, stream_path=0, video_output=True,
                              labels_on_vid=MODEL_OUTPUT_LABELS, verbose=USE_VERBOSE_ON_HVC).start()

    signal.signal(signal.SIGINT, hvc.stop)

    drone = Tello()
    print("Connecting ...")
    drone.connect()
    print("Connected")
    # time.sleep(5)
    print("Takeoff !")
    drone.takeoff()

    print(f"Battery level : {drone.get_battery()}")

    while hvc.is_running():
        state = hvc.get_prediction()
        drone_commands(drone, state)


def drone_cam_main():
    hvc = DroneHandVideoClassifier(model_path=MODEL_PATH, stream_path=0, video_output=True,
                                   labels_on_vid=MODEL_OUTPUT_LABELS, verbose=USE_VERBOSE_ON_HVC).start()

    signal.signal(signal.SIGINT, hvc.stop)

    print("T1")
    hvc.set_command(100)
    print("T1")

    while hvc.is_running():
        if hvc.get_command() == -1:
            state = hvc.get_prediction()
            hvc.set_command(state)
            time.sleep(PREDICTION_DELAY)


if __name__ == '__main__':
    drone_cam_main()
