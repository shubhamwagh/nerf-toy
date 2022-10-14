import tensorflow as tf


def psnr(y_true, y_pred):
    """
    Peak Signal to Noise ratio
    """
    return tf.image.psnr(y_true, y_pred, 1.0, name='PSNR')


def ssim(y_true, y_pred):
    """
    Structured similarity index
    """
    return tf.image.ssim(y_true, y_pred, max_val=1.0, filter_size=11,
                         filter_sigma=1.5, k1=0.01, k2=0.03)
