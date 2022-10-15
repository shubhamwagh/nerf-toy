import unittest
import pathlib
import numpy as np
from nerf_toy.data_loader import DataLoader
from nerf_toy.transforms import BasicEncodingTransform, PositionalEncodingTransform, GaussianFourierFeatureTransform

current_path = pathlib.Path(__file__).parent.resolve()


class TestDataLoader(unittest.TestCase):
    IMG_PATH = current_path.parent.joinpath('data', 'lion_face.jpg').as_posix()

    def test_data_loader_without_transform(self):
        loader = DataLoader(uri=self.IMG_PATH)
        yx_grid, target = loader.load_data()

        # image shape
        self.assertEqual(loader.img.shape, (256, 256, 3))

        # no transform
        self.assertIsNone(loader.transform)

        # shape 4D input and target
        self.assertEqual(len(yx_grid.shape), 4)
        self.assertEqual(len(target.shape), 4)

        # yx_grid and target have same height, width
        self.assertEqual(yx_grid.shape[1:-1], (256, 256))
        self.assertEqual(target.shape[1:-1], (256, 256))

        # yx_grid num_channel is 2 w/o encoding
        self.assertEqual(yx_grid.shape[-1], 2)

        # target num_channel is 3
        self.assertEqual(target.shape[-1], 3)

        # dtype is np.ndarray
        self.assertTrue(isinstance(yx_grid, np.ndarray))
        self.assertTrue(isinstance(target, np.ndarray))

        # type is float32
        self.assertEqual(yx_grid.dtype, 'float32')
        self.assertEqual(target.dtype, 'float32')

        # range values of input and target
        self.assertAlmostEqual(yx_grid.min(initial=0), 0, 2)
        self.assertAlmostEqual(yx_grid.max(initial=0), 1, 2)

        # range values of target
        self.assertEqual(target.min(initial=0), 0)
        self.assertEqual(target.max(initial=0), 1)

    def test_data_loader_with_basic_transform(self):
        transform = BasicEncodingTransform()
        loader = DataLoader(uri=self.IMG_PATH, transform=transform)
        yx_grid, target = loader.load_data()

        # image shape
        self.assertEqual(loader.img.shape, (256, 256, 3))

        # transform available
        self.assertIsNotNone(loader.transform)

        # shape 4D input and target
        self.assertEqual(len(yx_grid.shape), 4)
        self.assertEqual(len(target.shape), 4)

        # yx_grid and target have same height, width
        self.assertEqual(yx_grid.shape[1:-1], (256, 256))
        self.assertEqual(target.shape[1:-1], (256, 256))

        # yx_grid num_channel is 2 * 2, sin, cos components for two channels
        self.assertEqual(yx_grid.shape[-1], 4)

        # target num_channel is 3
        self.assertEqual(target.shape[-1], 3)

        # dtype is np.ndarray
        self.assertTrue(isinstance(yx_grid, np.ndarray))
        self.assertTrue(isinstance(target, np.ndarray))

        # type is float32
        self.assertEqual(yx_grid.dtype, 'float32')
        self.assertEqual(target.dtype, 'float32')

        # range values of input after encoding will be in [-1, 1]
        self.assertAlmostEqual(yx_grid.min(initial=0), -1, 2)
        self.assertAlmostEqual(yx_grid.max(initial=0), 1, 2)

        # range values of target
        self.assertEqual(target.min(initial=0), 0)
        self.assertEqual(target.max(initial=0), 1)

    def test_data_loader_with_positional_encoding_transform(self):
        L = 10
        transform = PositionalEncodingTransform(multires=L)
        loader = DataLoader(uri=self.IMG_PATH, transform=transform)
        yx_grid, target = loader.load_data()

        # image shape
        self.assertEqual(loader.img.shape, (256, 256, 3))

        # transform available
        self.assertIsNotNone(loader.transform)

        # shape 4D input and target
        self.assertEqual(len(yx_grid.shape), 4)
        self.assertEqual(len(target.shape), 4)

        # yx_grid and target have same height, width
        self.assertEqual(yx_grid.shape[1:-1], (256, 256))
        self.assertEqual(target.shape[1:-1], (256, 256))

        # yx_grid num_channel is L * 2 * 2, sin, cos components for two channels for all 10 frequencies
        self.assertEqual(yx_grid.shape[-1], 40)

        # target num_channel is 3
        self.assertEqual(target.shape[-1], 3)

        # dtype is np.ndarray
        self.assertTrue(isinstance(yx_grid, np.ndarray))
        self.assertTrue(isinstance(target, np.ndarray))

        # type is float32
        self.assertEqual(yx_grid.dtype, 'float32')
        self.assertEqual(target.dtype, 'float32')

        # range values of input after encoding will be in [-1, 1]
        self.assertAlmostEqual(yx_grid.min(initial=0), -1, 2)
        self.assertAlmostEqual(yx_grid.max(initial=0), 1, 2)

        # range values of target
        self.assertEqual(target.min(initial=0), 0)
        self.assertEqual(target.max(initial=0), 1)

    def test_data_loader_with_gaussian_fourier_feature_transform(self):
        mapping_size = 50
        scale = 10
        transform = GaussianFourierFeatureTransform(num_input_channels=2, mapping_size=mapping_size, scale=scale)
        loader = DataLoader(uri=self.IMG_PATH, transform=transform)
        yx_grid, target = loader.load_data()

        # image shape
        self.assertEqual(loader.img.shape, (256, 256, 3))

        # transform available
        self.assertIsNotNone(loader.transform)

        # shape 4D input and target
        self.assertEqual(len(yx_grid.shape), 4)
        self.assertEqual(len(target.shape), 4)

        # yx_grid and target have same height, width
        self.assertEqual(yx_grid.shape[1:-1], (256, 256))
        self.assertEqual(target.shape[1:-1], (256, 256))

        # yx_grid num_channel is mapping_size * 2 channels
        self.assertEqual(yx_grid.shape[-1], mapping_size * 2)

        # target num_channel is 3
        self.assertEqual(target.shape[-1], 3)

        # dtype is np.ndarray
        self.assertTrue(isinstance(yx_grid, np.ndarray))
        self.assertTrue(isinstance(target, np.ndarray))

        # type is float32
        self.assertEqual(yx_grid.dtype, 'float32')
        self.assertEqual(target.dtype, 'float32')

        # range values of input after encoding will be in [-1, 1]
        self.assertAlmostEqual(yx_grid.min(initial=0), -1, 2)
        self.assertAlmostEqual(yx_grid.max(initial=0), 1, 2)

        # range values of target
        self.assertEqual(target.min(initial=0), 0)
        self.assertEqual(target.max(initial=0), 1)


if __name__ == "__main__":
    unittest.main()
