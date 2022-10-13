import os
import imageio
import numpy as np
import tensorflow as tf
from nerf_toy.utils import to_image


class NerfToyPredictionSaverCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_test, y_test, every: int = 1, video_path='/tmp'):
        super(NerfToyPredictionSaverCallback, self).__init__()
        self.x_test = x_test
        self.y_test = y_test
        self.every = every
        self.prediction_imgs = []  # uint8
        self.video_path = video_path

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.every == 0:
            print('Evaluating Model...')
            pred = self.model(self.x_test, training=False)
            self.prediction_imgs.append(to_image(pred.numpy()[0]))

    def on_train_end(self, logs=None):
        print('Savin training convergence video....')
        data8 = np.stack(self.prediction_imgs)
        file_path = os.path.join(self.video_path, 'training_convergence.mp4')
        imageio.mimwrite(file_path, data8, fps=20)
