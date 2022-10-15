import unittest
import pathlib
import numpy as np
from nerf_toy.utils import read_img, normalise, arr_to_image

current_path = pathlib.Path(__file__).parent.resolve()


class TestUtils(unittest.TestCase):
    IMG_PATH = current_path.parent.joinpath('data', 'lion_face.jpg').as_posix()

    def test_read_img(self):
        rgb_img = read_img(self.IMG_PATH, rgb=True)
        self.assertEqual(rgb_img.shape, (256, 256, 3))

        bgr_img = read_img(self.IMG_PATH, rgb=False)
        self.assertEqual(bgr_img.shape, (256, 256, 3))

        np.testing.assert_array_equal(rgb_img[:, :, 0], bgr_img[:, :, -1])
        np.testing.assert_array_equal(rgb_img[:, :, -1], bgr_img[:, :, 0])

    def test_normalise(self):
        img = read_img(self.IMG_PATH)
        self.assertTrue(img.dtype == 'uint8')
        self.assertEqual(img.min(), 0)
        self.assertEqual(img.max(), 255)

        norm_arr = normalise(img)
        self.assertTrue(norm_arr.dtype == 'float32')
        self.assertEqual(norm_arr.min(), 0.0)
        self.assertEqual(norm_arr.max(), 1.0)

    def test_to_image(self):
        norm_arr = normalise(read_img(self.IMG_PATH))
        converted_img = arr_to_image(norm_arr)
        self.assertTrue(converted_img.dtype == 'uint8')
        self.assertEqual(converted_img.min(), 0)
        self.assertEqual(converted_img.max(), 255)


if __name__ == "__main__":
    unittest.main()
