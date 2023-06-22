import cv2
import numpy as np
import os
import tensorflow as tf
from keras import layers, models
import mediapipe as mp
import argparse


def create_model(n_classes):
    model = models.Sequential([
        layers.Dense(63, activation='relu', input_shape=(63,)),
        layers.Dropout(0.2),
        layers.Dense(300, activation='relu', input_shape=(300,)),
        layers.Dropout(0.2),
        layers.Dense(500, activation='relu', input_shape=(500,)),
        layers.Dropout(0.2),
        layers.Dense(300, activation='relu', input_shape=(300,)),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation="softmax")
    ])

    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.metrics.SparseCategoricalAccuracy()])

    return model


def load_data_from_file(path: str, class_id: int):
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        with open(path, 'r') as f:
            lm_from_csv = np.loadtxt(f.readlines(), delimiter=',').reshape((-1, 63))
            labels = np.ones(lm_from_csv.shape[0], dtype=int) * class_id
        return lm_from_csv, labels
    else:
        raise ValueError(f"File <{path}> is empty or doesn't exist")


def main():
    parser = argparse.ArgumentParser(description="Train and visualize")
    parser.add_argument('--no_train',
                        help="Disable model training and use pre-existing model",
                        action="store_true")
    args = parser.parse_args()

    data_path = "Assets/datasets_records"
    train_model = not args.no_train
    model_path = "Assets/model_data/"
    files_names = [class_file for class_file in os.listdir(data_path)]

    if train_model:
        x, y = load_data_from_file(data_path + "/" + files_names[0], class_id=0)
        for i in range(1, len(files_names)):
            xc, yc = load_data_from_file(data_path + "/" + files_names[i], class_id=i)
            x = np.concatenate((x, xc))
            y = np.concatenate((y, yc))

        model = create_model(len(files_names))
        model.summary()

        model.fit(x=x, y=y,
                  epochs=20)
        model.save(model_path)
    else:
        model = models.load_model(model_path)

    posture_list = [x[:-4] for x in files_names]
    print(f"INFO: Loaded classes : {posture_list}")

    # Capture object initialisation
    cap = cv2.VideoCapture(0)

    # Hands detection objects
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.9)

    while cap.isOpened():
        ret, img = cap.read()

        if ret:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, c = img_rgb.shape

            # Detection
            results = hands.process(img_rgb)

            nb_hands = 0
            # if at least one hand is detected :
            if results.multi_hand_landmarks:
                nb_hands = len(results.multi_hand_landmarks)

            if nb_hands == 0:
                cv2.rectangle(img_rgb, (0, 0), (w, h), (0, 0, 0), 5)
                cv2.putText(img_rgb, "No hands detected",
                            org=(10, 20),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale=1, color=(100, 0, 255), thickness=1)
            elif nb_hands == 1:
                cv2.rectangle(img_rgb, (0, 0), (w, h), (0, 255, 0), 5)
                coords_list = np.zeros((21, 3), dtype=np.float32)
                for handLms in results.multi_hand_landmarks:
                    # Landmarks enumeration
                    for nb, lm in enumerate(handLms.landmark):
                        coords_list[nb, :] = [lm.x, lm.y, lm.z]

                if coords_list.flatten().shape == (63,):
                    pred = model.predict(coords_list.reshape(1, 63), verbose=False)
                    for c in range(len(posture_list)):
                        cv2.rectangle(img=img_rgb,
                                      pt1=(0, 50 + 50 * c),
                                      pt2=(int(200 * pred[0, c]), 50 * c),
                                      color=(int(255 * (1 - pred[0, c])), int(255 * pred[0, c]), 0),
                                      thickness=-1)
                        cv2.rectangle(img=img_rgb,
                                      pt1=(0, 50 + 50 * c),
                                      pt2=(int(200), 50 * c),
                                      color=(100, 100, 100),
                                      thickness=1)
                        cv2.putText(img_rgb, posture_list[c],
                                    (10, 25 + 50 * c),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (55, 0, 200), 1)
            else:
                cv2.rectangle(img_rgb, (0, 0), (w, h), (255, 0, 0), 5)
                cv2.putText(img_rgb, "Too many hands !",
                            (w // 10, h // 2),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            cv2.imshow("out", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == '__main__':
    main()
