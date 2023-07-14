import cv2
import numpy as np


class HMI:
    def __init__(self,
                 window_name: str = "HMI"):
        self.window_name = window_name
        self.objects = []
        self.hmi_output = np.zeros((480, 640, 3), dtype=np.uint8)
        self.hand_coords = [[0, 0],
                            [0, 0]]

    def draw(self):
        self.hmi_output = np.zeros((480, 640, 3))

        for obj in self.objects:
            obj.draw(self.hmi_output)

        self.draw_hands()

        cv2.imshow("HMI", self.hmi_output)

    def draw_hands(self):
        cv2.circle(self.hmi_output,
                   center=self.hand_coords[0],
                   radius=10,
                   color=(0.5, 0.5, 0),
                   thickness=3,
                   lineType=cv2.LINE_4)

        cv2.circle(self.hmi_output,
                   center=self.hand_coords[1],
                   radius=10,
                   color=(0, 0.5, 0.5),
                   thickness=3,
                   lineType=cv2.LINE_4)

    def set_hands_coords(self, coords):
        self.hand_coords = coords

    def add_object(self, obj):
        self.objects.append(obj)

    def delete_object(self, obj_id) -> None:
        self.objects.pop(obj_id)


class Element:
    def __init__(self,
                 position: list | np.ndarray,
                 hit_box_dims: list | tuple = (20, 20),
                 color: tuple | list | np.ndarray = (1, 1, 1),
                 can_by_grabbed: bool = True):
        self.hit_box = hit_box_dims
        self.position = position
        self.color = color
        self.can_by_grabbed = can_by_grabbed
        self.grabbed = False
        self.grabbed_by = 0

    def get_hit_box(self) -> (tuple | list):
        return self.hit_box

    def set_position(self, position: list | np.ndarray):
        self.position = position

    def get_position(self) -> list:
        return self.position

    def set_grabbed(self, grabbed: bool, holder_index: int = 0):
        if self.can_by_grabbed:
            self.grabbed = grabbed
            self.grabbed_by = holder_index

    def is_grabbed(self):
        return self.grabbed


class Ball(Element):
    def __init__(self, position: list | np.ndarray, ball_radius: int = 30,
                 color: tuple | list | np.ndarray = (1, 1, 1)):
        super().__init__(position=position,
                         color=color,
                         hit_box_dims=(ball_radius, ball_radius))
        self.ball_radius = ball_radius

    def draw(self, src):
        cv2.circle(src,
                   center=self.position,
                   radius=self.ball_radius,
                   color=tuple(self.color),
                   thickness=-1)


class Box(Element):
    def __init__(self, position: list | np.ndarray,
                 box_size: tuple | list | np.ndarray = (100, 40),
                 color: tuple | list | np.ndarray = (1, 1, 1)):
        super().__init__(position=position,
                         color=color,
                         hit_box_dims=[box_size[0] // 2, box_size[1] // 2])
        self.box_size = box_size

    def draw(self, src):
        cv2.rectangle(src,
                      pt1=(self.position[0] - self.box_size[0] // 2, self.position[1] - self.box_size[1] // 2),
                      pt2=(self.position[0] + self.box_size[0] // 2, self.position[1] + self.box_size[1] // 2),
                      color=self.color,
                      thickness=-1,
                      lineType=cv2.LINE_4)
