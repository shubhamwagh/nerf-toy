import cv2
import numpy as np
import tensorflow as tf


def read_img(img_path: str, rgb: bool = True) -> np.ndarray:
    img = cv2.imread(img_path)
    if rgb:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        return img


def to_image(img_arr: np.ndarray) -> np.ndarray:
    # img_arr have values in range [0, 1]
    img_arr = np.clip(img_arr, 0.0, 1.0)
    img_arr = img_arr * 255.0
    return img_arr.astype('uint8')


def normalise(img: np.ndarray) -> np.ndarray:
    # img is uint8 numpy array in range [0, 255]
    img_arr = img.astype('float32')
    img_arr = img_arr / 255.0
    return img_arr
