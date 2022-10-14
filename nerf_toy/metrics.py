import tensorflow as tf


def PSNR(max_value: float = 1.0):
    def psnr(y_true, y_pred):
        """
        Peak Signal to Noise ratio
        """
        return tf.image.psnr(y_true, y_pred, max_val=max_value, name='PSNR')

    return psnr


def SSIM(max_value: float = 1.0):
    def ssim(y_true, y_pred):
        """
        Structured similarity index
        """
        return tf.image.ssim(y_true, y_pred, max_val=max_value, filter_size=11,
                             filter_sigma=1.5, k1=0.01, k2=0.03)

    return ssim
