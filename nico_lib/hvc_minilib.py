from multiprocessing import Process, Value
import numpy as np
import cv2
import mediapipe as mp
from keras import models


class HandVideoClassifier:
    def __init__(self,
                 model_path: str,
                 stream_path: int | str = 0,
                 video_output: bool = False,
                 verbose: bool = False,
                 labels_on_vid: list | np.ndarray = None,
                 always_on_top: bool = True) -> None:
        """
        Description

        HandVideoClassifier
        --------
        Video Classifier based on Mediapipe hand landmarks using
        a given TensorFlow model and giving a real time prediction.

        :param model_path: folder that contains the TensorFlow Model,
        :param stream_path: integer for camera usage (0 for main camera), string for video file,
        :param video_output: enables OpenCV video output,
        :param verbose: enables verbose mode,
        :param labels_on_vid: list or array of labels to show on video output,
        :param always_on_top: keeps video output in front of other apps.
        """
        self.__process = None
        self.__stream = None
        self.__prediction_left = Value("i", -1)
        self.__prediction_right = Value("i", -1)
        self.left_center_coords_x = Value("i", -1)
        self.left_center_coords_y = Value("i", -1)
        self.right_center_coords_x = Value("i", -1)
        self.right_center_coords_y = Value("i", -1)
        self.__running = Value('i', 0)
        self.__video_output = video_output
        self.__verbose = verbose
        self.__stream_path = stream_path
        self.model_path = model_path
        self.labels = labels_on_vid
        self.always_on_top = always_on_top

    def start(self) -> "HandVideoClassifier":
        """
        Detection process startup.

        :return: HandVideoClassifier object
        """
        if self.__verbose:
            print("INFO: Starting capture and detection ...")
        self.__process = Process(target=self._mainloop_subprocess, args=(self.model_path, self.labels))
        self.__process.start()

        while not self.is_running():
            pass

        if self.__verbose:
            print("INFO: Process has started")

        return self

    def _mainloop_subprocess(self, model_path, labels):
        # Hands detection objects
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(min_detection_confidence=0.9, max_num_hands=2)

        # Model loading
        model = models.load_model(model_path)

        # Capture type definition
        if type(self.__stream_path) == int:
            self.__stream = cv2.VideoCapture(self.__stream_path, cv2.CAP_DSHOW)
        else:
            self.__stream = cv2.VideoCapture(self.__stream_path)

        if labels is not None and model.layers[-1].output_shape[1] == len(labels):
            self.labels = labels
        else:
            raise ValueError("The labels list must be of length {0}".format((model.layers[-1]).output_shape[1]))

        # Allowing main process to continue and finishing startup.
        self._set_running()

        # Main loop, stops when EOF, escape or user code asking detection shutdown.
        while self.__running and self.__stream.isOpened():
            grabbed, src = self.__stream.read()
            if not grabbed:
                self.stop()
            else:
                src = cv2.flip(src, 1)
                rgb_src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

                # Detection
                results = hands.process(rgb_src)
                if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
                    coords_list = np.zeros((2, 21, 3), dtype=np.float32)
                    for handLms, hand_id in zip(results.multi_hand_landmarks, [0, 1]):
                        # Landmarks enumeration
                        for nb, lm in enumerate(handLms.landmark):
                            coords_list[hand_id, nb, :] = [lm.x, lm.y, lm.z]

                    pred_left = model.predict(coords_list[0, :, :].reshape(1, 63), verbose=False)
                    self.__prediction_left.value = np.argmax(pred_left)

                    pred_right = model.predict(coords_list[1, :, :].reshape(1, 63), verbose=False)
                    self.__prediction_right.value = np.argmax(pred_right)

                    self.left_center_coords_x.value = int(results.multi_hand_landmarks[0].landmark[9].x \
                                                          * rgb_src.shape[1])
                    self.left_center_coords_y.value = int(results.multi_hand_landmarks[0].landmark[9].y \
                                                          * rgb_src.shape[0])
                    self.right_center_coords_x.value = int(results.multi_hand_landmarks[1].landmark[9].x \
                                                           * rgb_src.shape[1])
                    self.right_center_coords_y.value = int(results.multi_hand_landmarks[1].landmark[9].y \
                                                           * rgb_src.shape[0])
                else:
                    self.__prediction_left.value = -1
                    self.__prediction_right.value = -1

                if self.__video_output:
                    if self.labels:
                        if self.__prediction_left.value != -1:
                            cv2.putText(src, self.labels[self.__prediction_left.value],
                                        org=(self.left_center_coords_x.value, self.left_center_coords_y.value),
                                        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        fontScale=1, color=(255, 255, 255), thickness=1)
                        if self.__prediction_right.value != -1:
                            cv2.putText(src, self.labels[self.__prediction_right.value],
                                        org=(self.right_center_coords_x.value, self.right_center_coords_y.value),
                                        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        fontScale=1, color=(255, 255, 255), thickness=1)
                    cv2.imshow("Video Output", src)
                    if self.always_on_top:
                        cv2.setWindowProperty("Video Output", cv2.WND_PROP_TOPMOST, 1)
                    if cv2.waitKey(1) & 0xFF == 27:
                        self.stop()

    def get_prediction(self) -> tuple:
        """
        Returns the argmax of the classifier output for both hands

        :return: Classifier Output, -1 if no class was detected.
        """
        return self.__prediction_left.value, self.__prediction_right.value

    def get__hands_coords(self) -> list:
        """
        Returns the coordinates in the image for both hands

        :return: Hands coords in frame.
        """
        return [[self.left_center_coords_x.value, self.left_center_coords_y.value],
                [self.right_center_coords_x.value, self.right_center_coords_y.value]]

    def is_running(self) -> int:
        """
        Returns the running state of the detection process.

        :return: Running state
        """
        return self.__running.value == 1  # necessary due to subprocess integer shared values limitations.

    def stop(self):
        """
        Stops the running process.
        """
        if self.__verbose:
            print("INFO: Shutting down process")

        if not self.is_running():
            raise Exception("Cannot stop : no process is running")

        self.__running.value = 0

        if self.__video_output:
            self.__stream.release()
            cv2.destroyAllWindows()

        if self.__verbose:
            print("INFO: Stream stopped")

        if self.__verbose:
            print("INFO: Process terminated")

    def _set_running(self):
        self.__running.value = 1


if __name__ == '__main__':
    pass
