import numpy as np
from nerf_toy.utils import read_img, pre_process

from typing import Tuple, Callable, Optional


class DataLoader(object):
    def __init__(self, img_path: str, transform: Optional[Callable] = None):
        self.img = read_img(img_path)
        self.transform = transform

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads normalised pixel coordinates grid and normalised target
        """
        target = np.expand_dims(pre_process(self.img), 0)
        batch, height, width, channels = target.shape
        yx_grid = self._create_xy_grid(grid_size=(height, width))

        assert len(yx_grid.shape) == len(target.shape)

        if self.transform is not None:
            return self.transform(yx_grid), target
        else:
            return yx_grid, target

    @staticmethod
    def _create_xy_grid(grid_size: Tuple) -> np.ndarray:
        """
        Creates mesh grid of normalised pixel coordinates based on matrix indexing convention
        """
        h, w = grid_size
        coords_i = np.linspace(0, 1, h, endpoint=False)
        coords_j = np.linspace(0, 1, w, endpoint=False)
        grid = np.stack(np.meshgrid(coords_i, coords_j, indexing='ij'), axis=-1).astype('float32')
        return grid
