import cv2
import numpy as np


def read_img(img_path: str, rgb: bool = True) -> np.ndarray:
    """
    Reads file path to output uint8 image
    """
    img = cv2.imread(img_path)
    if rgb:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        return img


def arr_to_image(img_arr: np.ndarray) -> np.ndarray:
    """
    Converts from normalised array to uint8 image
    """
    img_arr = np.clip(img_arr, 0.0, 1.0)
    img_arr = img_arr * 255.0
    return img_arr.astype('uint8')


def normalise(img: np.ndarray) -> np.ndarray:
    """
    Converts from uint8 image to normalise numpy array
    """
    img_arr = img.astype('float32')
    img_arr = img_arr / 255.0
    return img_arr
