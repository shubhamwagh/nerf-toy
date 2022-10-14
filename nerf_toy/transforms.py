import numpy as np


class BasicEncodingTransform(object):
    """
    Simply wraps input coordinates around the circle.
    https://arxiv.org/abs/2003.08934
    https://arxiv.org/abs/2006.10739
    """

    def __init__(self):
        super(BasicEncodingTransform, self).__init__()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 4, 'Expected 4D input (got {}D input)'.format(len(x.shape))
        x_proj = 2 * np.pi * x
        return np.concatenate((np.sin(x_proj), np.cos(x_proj)), axis=-1).astype('float32')


class PositionalEncodingTransform(object):
    """
    Deterministic mapping containing on-axis log-linear spaced frequencies for each dimension.
    https://arxiv.org/abs/2003.08934
    https://arxiv.org/abs/2006.10739
    """

    def __init__(self, multires: int = 10):
        super(PositionalEncodingTransform, self).__init__()
        self.num_freq = multires
        self._max_log2_freq = multires - 1
        self._freq_bands = 2 ** np.linspace(0, self._max_log2_freq, self.num_freq)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 4, 'Expected 4D input (got {}D input)'.format(len(x.shape))

        num_out_channels = self.num_freq * 2 * 2
        pos_enc = np.zeros((x.shape[0], x.shape[1], x.shape[2], num_out_channels))
        channel_id = 0
        for freq in self._freq_bands:
            x_proj = np.pi * freq * x
            pos_enc[:, :, :, channel_id: channel_id + 4] = np.concatenate((np.sin(x_proj), np.cos(x_proj)), axis=-1)
            channel_id += 4
        return pos_enc.astype('float32')


class GaussianFourierFeatureTransform(object):
    """
    An implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    Given an input of size [batch, height, width, channels],
    returns a tensor of size [batch, height, width, mapping_size*2].
    """

    def __init__(self, num_input_channels: int, mapping_size: int = 256, scale: int = 10):
        super(GaussianFourierFeatureTransform, self).__init__()
        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = np.random.randn(num_input_channels, mapping_size) * scale

    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 4, 'Expected 4D input (got {}D input)'.format(len(x.shape))

        batches, height, width, channels = x.shape

        assert channels == self._num_input_channels, \
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, H, W, C] to [(B*H*W), C].
        x = x.reshape(batches * height * width, channels)

        x = x @ self._B

        # From [(B*H*W), C] to [B, H, W, C]
        x = x.reshape(batches, height, width, self._mapping_size)

        x = 2 * np.pi * x
        return np.concatenate((np.sin(x), np.cos(x)), axis=-1).astype('float32')
