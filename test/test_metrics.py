import unittest
import pathlib
import tensorflow as tf
from nerf_toy.metrics import PSNR, SSIM

current_path = pathlib.Path(__file__).parent.resolve()


class TestMetrics(unittest.TestCase):
    IMG_PATH_1 = current_path.parent.joinpath('data', 'lion_face.jpg').as_posix()
    IMG_PATH_2 = current_path.parent.joinpath('data', 'noisy_lion_face.jpg').as_posix()

    def _read_image_tf(self, img_path):
        with open(img_path, 'rb') as file:
            image = tf.image.decode_image(file.read())
        return image

    def test_psnr(self):
        img1 = self._read_image_tf(self.IMG_PATH_1)
        img2 = self._read_image_tf(self.IMG_PATH_2)

        psnr_255 = PSNR(max_value=255.0)
        self.assertAlmostEqual(psnr_255(img1, img2).numpy(), 17.21, 1)

        psnr_1 = PSNR(max_value=1.0)
        self.assertAlmostEqual(psnr_1(tf.image.convert_image_dtype(img1, tf.float32),
                                      tf.image.convert_image_dtype(img2, tf.float32)).numpy(), 17.21, 1)

    def test_ssim(self):
        img1 = self._read_image_tf(self.IMG_PATH_1)
        img2 = self._read_image_tf(self.IMG_PATH_2)

        ssim_255 = SSIM(max_value=255.0)
        self.assertAlmostEqual(ssim_255(img1, img2).numpy(), 0.85, 2)

        ssim_1 = SSIM(max_value=1.0)
        self.assertAlmostEqual(ssim_1(tf.image.convert_image_dtype(img1, tf.float32),
                                      tf.image.convert_image_dtype(img2, tf.float32)).numpy(), 0.85, 2)


if __name__ == "__main__":
    unittest.main()
