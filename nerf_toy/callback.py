import os
import imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from nerf_toy.utils import arr_to_image
from IPython.display import clear_output

from typing import List


class PredictionVideoSaverCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_test, y_test, every: int = 1, video_path='/tmp'):
        super(PredictionVideoSaverCallback, self).__init__()
        self.x_test = x_test
        self.y_test = y_test
        self.every = every
        self.prediction_imgs: List[np.uint8] = []  # uint8
        self.video_path = video_path
        self.saved_path = ''

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.every == 0:
            pred = self.model(self.x_test, training=False)
            self.prediction_imgs.append(arr_to_image(pred.numpy()[0]))

    def on_train_end(self, logs=None):
        print('Saving training convergence video.....')
        data8 = np.stack(self.prediction_imgs)

        count = 0
        while os.path.exists(os.path.join(self.video_path, f"training_convergence_{count}.mp4")):
            count += 1

        file_path = os.path.join(self.video_path, f'training_convergence_{count}.mp4')
        imageio.mimwrite(file_path, data8, fps=20)
        self.saved_path = file_path
        print(f'Saved training convergence video as {file_path}')


class PlotLossesAndMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(PlotLossesAndMetricsCallback, self).__init__()
        self.losses = []
        self.psnrs = []
        self.ssims = []
        self.epochs = []

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)
        self.losses.append(logs.get('loss'))
        self.psnrs.append(logs.get('psnr'))
        self.ssims.append(logs.get('ssim'))

        fig, ax = plt.subplots(1, 3, figsize=(18, 4), sharex=True)

        clear_output(wait=True)

        ax[0].set_title('Train loss')
        ax[0].plot(self.epochs, self.losses, label='train loss')
        ax[0].set_ylabel('loss')
        ax[0].set_xlabel('epochs')
        ax[0].grid(visible=True)

        ax[1].set_title('PSNR')
        ax[1].plot(self.epochs, self.psnrs, label='psnr')
        ax[1].set_ylabel('PSNR')
        ax[1].set_xlabel('epochs')
        ax[1].grid(visible=True)

        ax[2].set_title('SSIM')
        ax[2].plot(self.epochs, self.ssims, label='ssim')
        ax[2].set_ylabel('ssim')
        ax[2].set_xlabel('epochs')
        ax[2].grid(visible=True)

        plt.show()
