import unittest
import numpy as np
from nerf_toy.transforms import BasicEncodingTransform, PositionalEncodingTransform, GaussianFourierFeatureTransform


class TestTransforms(unittest.TestCase):
    def test_basic_encoding_transform(self):
        input_shape = (1, 100, 100, 2)
        x = np.random.randn(*input_shape)
        encoding = BasicEncodingTransform()
        x_enc = encoding(x)
        expected_out_channels = 2 * 2

        # after encoding, x will have 4 channels -> sin, cos components for two channels
        self.assertEqual(x_enc.shape, input_shape[:-1] + (expected_out_channels,))
        self.assertTrue(isinstance(x_enc, np.ndarray))
        self.assertEqual(x_enc.dtype, 'float32')
        self.assertAlmostEqual(x_enc.min(initial=0), -1, 3)
        self.assertAlmostEqual(x_enc.max(initial=0), 1, 3)

        # shape must be 4D input
        with self.assertRaises(AssertionError):
            incorrect_input_shape = (100, 100, 2)
            incorrect_x = np.random.randn(*incorrect_input_shape)
            encoding(incorrect_x)

    def test_positional_encoding_transform(self):
        input_shape = (1, 100, 100, 2)
        x = np.random.randn(*input_shape)
        L = 10
        encoding = PositionalEncodingTransform(multires=L)
        x_enc = encoding(x)
        expected_out_channels = 10 * 2 * 2

        # after encoding, x will have 10 * 4 channels -> sin, cos components for two channels for all 10 frequencies
        self.assertEqual(encoding.num_freq, L)
        self.assertEqual(x_enc.shape, input_shape[:-1] + (expected_out_channels,))
        self.assertTrue(isinstance(x_enc, np.ndarray))
        self.assertEqual(x_enc.dtype, 'float32')
        self.assertAlmostEqual(x_enc.min(initial=0), -1, 3)
        self.assertAlmostEqual(x_enc.max(initial=0), 1, 3)

        # shape must be 4D input
        with self.assertRaises(AssertionError):
            incorrect_input_shape = (100, 100, 2)
            incorrect_x = np.random.randn(*incorrect_input_shape)
            encoding(incorrect_x)

    def test_gaussian_fourier_feature_transform(self):
        input_shape = (1, 100, 100, 2)
        x = np.random.randn(*input_shape)
        num_input_channels = x.shape[-1]
        mapping_size = 50
        scale = 10
        encoding = GaussianFourierFeatureTransform(num_input_channels=num_input_channels,
                                                   mapping_size=mapping_size,
                                                   scale=scale)
        x_enc = encoding(x)
        expected_out_channels = mapping_size * 2

        # after encoding, x will have mapping_size * 2 channels
        self.assertEqual(x_enc.shape, input_shape[:-1] + (expected_out_channels,))
        self.assertTrue(isinstance(x_enc, np.ndarray))
        self.assertEqual(x_enc.dtype, 'float32')
        self.assertAlmostEqual(x_enc.min(initial=0), -1, 3)
        self.assertAlmostEqual(x_enc.max(initial=0), 1, 3)

        # shape must be 4D input
        with self.assertRaises(AssertionError):
            incorrect_input_shape = (100, 100, 2)
            incorrect_x = np.random.randn(*incorrect_input_shape)
            encoding(incorrect_x)

        # incorrect num_input_channels
        with self.assertRaises(AssertionError):
            incorrect_num_channels = 3
            incorrect_encoding = GaussianFourierFeatureTransform(num_input_channels=incorrect_num_channels,
                                                                 mapping_size=mapping_size,
                                                                 scale=scale)
            incorrect_encoding(x)


if __name__ == "__main__":
    unittest.main()
