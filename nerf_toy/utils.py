import imageio
import numpy as np


def read_img(uri: str, rgb: bool = True) -> np.ndarray:
    """
    Reads file path to output uint8 image
    """
    img = imageio.imread(uri)
    if rgb:
        return img
    else:
        return img[:, :, ::-1]


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
