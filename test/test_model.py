import unittest

from nerf_toy.model import base_model


class TestModel(unittest.TestCase):
    def test_base_model(self):
        input_shape = (100, 100, 3)
        model = base_model(input_shape=input_shape, output_dim=input_shape[-1], num_layers=4, num_channels=16)

        self.assertEqual(model.count_params(), 851)
        self.assertEqual(model.input_shape[1:], input_shape)
        self.assertEqual(model.output_shape[1:], input_shape)


if __name__ == "__main__":
    unittest.main()
