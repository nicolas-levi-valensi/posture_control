from multiprocessing import Process, Value
import numpy as np
import cv2
import mediapipe as mp
from keras import models


class HandVideoClassifier:
    def __init__(self,
                 model_path: str,
                 stream_path: int | str = 0,
                 video_output: bool = False, verbose: bool = False,
                 labels_on_vid: list | np.ndarray = None) -> None:
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
        :param labels_on_vid: list or array of labels to show on video output.
        """
        self.__process = None
        self.__stream = None
        self.__prediction = Value("i", -1)
        self.__running = Value('i', 0)
        self.__video_output = video_output
        self.__verbose = verbose
        self.__stream_path = stream_path
        self.model_path = model_path
        self.labels = labels_on_vid

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
        hands = mp_hands.Hands(min_detection_confidence=0.9)

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

        # Allowing main process to continue and finishing startup
        self._set_running()

        # Main loop, stops when EOF, escape or user code asking detection shutdown
        while self.__running and self.__stream.isOpened():
            grabbed, src = self.__stream.read()
            if not grabbed:
                self.stop()
            else:
                rgb_src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

                # Detection
                results = hands.process(rgb_src)
                if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
                    coords_list = np.zeros((21, 3), dtype=np.float32)
                    for handLms in results.multi_hand_landmarks:
                        # Landmarks enumeration
                        for nb, lm in enumerate(handLms.landmark):
                            coords_list[nb, :] = [lm.x, lm.y, lm.z]

                    pred = model.predict(coords_list.reshape(1, 63), verbose=False)
                    self.__prediction.value = np.argmax(pred)
                else:
                    self.__prediction.value = -1

                if self.__video_output:
                    if self.labels:
                        if self.__prediction.value != -1:
                            cv2.putText(src, self.labels[self.__prediction.value],
                                        org=(20, 20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        fontScale=1, color=(255, 255, 255), thickness=1)
                    cv2.imshow("Video Output", src)
                    if cv2.waitKey(1) & 0xFF == 27:
                        self.stop()

    def get_prediction(self):
        """
        Returns the argmax of the classifier output.

        :return: Classifier Output, -1 if no class was detected.
        """
        return self.__prediction.value

    def is_running(self):
        """
        Returns the running state of the detection process.

        :return: Running state
        """
        return self.__running.value

    def stop(self):
        """
        Stops the running process.
        """
        if self.__verbose:
            print("INFO: Shutting down process")

        if not self.is_running():
            raise Exception("Cannot stop : no process is running")

        self.__running.value = 0

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
