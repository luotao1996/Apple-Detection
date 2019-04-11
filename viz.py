#!usr/bin/python
# -*- coding: utf-8 -*-
import cv2
from object_detection.utils.visualization_utils import draw_bounding_box_on_image_array

def draw_fruits_box(img_bgr, fruits):
    """
    Draw bounding box and distance of detected apple.
    :param img_bgr: Original image where apple are detected.
    :param fruits: a list of fruits objects.
    :return A numpy
    """
    if len(fruits) == 0:
        return img_bgr

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # img_pil = Image.fromarray(img_rgb)

    for fruit in fruits:
        if fruit.box is not None:
            xmin, ymin, xmax, ymax = fruit.box
            draw_bounding_box_on_image_array(img_rgb, ymin, xmin, ymax, xmax, color='white', thickness=2,
                                             display_str_list=[
                                                 ' Type: {} Distance:{:.2f}cm'.format(fruit.cls,fruit.distance)],
                                             use_normalized_coordinates=False)

    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

