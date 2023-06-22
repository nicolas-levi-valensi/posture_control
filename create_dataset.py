import os.path
import tkinter as tk
from tkinter.messagebox import askokcancel, showinfo
import cv2
import numpy as np
import csv
import mediapipe as mp

class_path = ""
show_lm = True


def change_inv_file():
    def edit():
        global class_path
        if entry.get() != "":
            if os.path.isfile(f"./datasets_records/{entry.get()}.csv"):
                valid_path = askokcancel(title="Overwrite ?",
                                         message=f"This will overwrite the data of the class : {entry.get()}.")
                if not valid_path:
                    return
            showinfo(title="Startup", message="The acquisition will begin in a few seconds")
            class_path = f"./datasets_records/{entry.get()}.csv"
            app.destroy()

    def on_close():
        exit(0)

    app = tk.Tk("Class name")

    label = tk.Label(master=app, text="Class name to register :", font='Helvetica 16 bold')
    label.place(relx=0.5, rely=0.2, anchor="n")

    frame = tk.Frame(master=app, background="#4f23a5",
                     width=300, height=100)
    entry = tk.Entry(master=frame,
                     width=30, font="Helvetica 16 bold", justify="center")
    entry.place(relx=0.5, rely=0.2, anchor="center")
    button = tk.Button(master=frame,
                       text="OK",
                       command=edit)
    button.place(relx=0.5, rely=0.7, anchor="center")
    frame.place(relx=0.5, rely=0.9, anchor="s")

    app.protocol("WM_DELETE_WINDOW", on_close)
    app.title(string="Class name chooser")
    app.geometry("400x300")
    app.config(bg='#4f23a5')

    app.mainloop()


def main():
    change_inv_file()

    # Hands detection objects
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.9)
    mp_draw = mp.solutions.drawing_utils

    # Capture object initialisation
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    count = 0
    with open(class_path, 'w', encoding='UTF8', newline='') as f:
        file_writer = csv.writer(f)
        while cap.isOpened():
            ret, img = cap.read()
            if ret:
                img_rgb = cv2.cvtColor(img[:img.shape[0] // 2, :img.shape[1] // 2], cv2.COLOR_BGR2RGB)
                h, w, c = img_rgb.shape

                # Detection
                results = hands.process(img_rgb)

                nb_hands = 0
                # if at least one hand is detected :
                if results.multi_hand_landmarks:
                    nb_hands = len(results.multi_hand_landmarks)

                if nb_hands == 0:
                    cv2.rectangle(img_rgb, (0, 0), (w, h), (0, 0, 0), 5)
                elif nb_hands == 1:
                    cv2.rectangle(img_rgb, (0, 0), (w, h), (0, 255, 0), 5)

                    for handLms in results.multi_hand_landmarks:

                        mp_draw.draw_landmarks(img_rgb, handLms, mp_hands.HAND_CONNECTIONS)

                        # Landmarks enumeration
                        coords_list = np.zeros((21, 3), dtype=np.float32)
                        for nb, lm in enumerate(handLms.landmark):
                            coords_list[nb, :] = [lm.x, lm.y, lm.z]
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            cv2.putText(img_rgb, str(nb), (cx + 5, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    if cv2.waitKey(1) & 0xFF == ord("a"):
                        file_writer.writerow(coords_list.flatten())
                        print(count)
                        count += 1

                else:
                    cv2.rectangle(img_rgb, (0, 0), (w, h), (255, 0, 0), 5)
                    cv2.putText(img_rgb, "Too many hands !",
                                (w // 10, h // 2),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

                img = img // 2
                img[:h, :w] = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow('img', img)
                cv2.setWindowProperty("img", cv2.WND_PROP_TOPMOST, 1)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()

    if show_lm:
        with open(class_path, 'r') as f:
            if os.path.getsize(class_path) > 0:
                lm_from_csv = np.loadtxt(f.readlines(), delimiter=',').reshape((-1, 21, 3))

        for i in range(lm_from_csv.shape[0]):
            im = np.ones((img_rgb.shape[0] * 5, img_rgb.shape[1] * 5), dtype=np.uint8) * 255
            for lm in lm_from_csv[i]:
                cv2.circle(img=im,
                           center=(int(lm[0] * im.shape[1]), int(lm[1] * im.shape[0])),
                           radius=3,
                           color=(100, int(lm[2] * 5 * 255) % 255, 100),
                           thickness=2)
            cv2.imshow(f"Landmarks in {class_path}", im)
            cv2.waitKey(1)


if __name__ == '__main__':
    main()
