import numpy as np
import tensorflow as tf
from utils.layer import *


class ScolioNet(object):

    def __init__(self, x, landmarks, is_training, verbose=False):
        self.x = x
        self.landmarks = landmarks
        self.is_training = is_training
        self.predictions = self.build_model(x, self.is_training, verbose=verbose)

    def build_model(self, x, is_training, verbose):
        print("+---------------------------------------------------------------------------------------+")
        print("|                            Building the model -- ScolioNet                            |")
        print("+---------------------------------------------------------------------------------------+")
        print("|                                                                                       |")
        with tf.variable_scope("scolionet"):
            with tf.variable_scope("conv2d_0_1"):
                x = Conv2d(x, out_channels=64, kernel_size=3, strides=1, padding="SAME", name="conv", verbose=verbose)
            self.block1 = x
            with tf.variable_scope("conv2d_1"):
                x = BatchNorm(x, name="batchnorm", training=is_training, verbose=verbose)
                x = LeakyReLU(x, name="relu", verbose=verbose)
                x = Conv2d(x, out_channels=64, kernel_size=3, strides=1, padding="SAME", name="conv", verbose=verbose)
            with tf.variable_scope("conv2d_2"):
                x = BatchNorm(x, name="batchnorm", training=is_training, verbose=verbose)
                x = LeakyReLU(x, name="relu", verbose=verbose)
                x = Conv2d(x, out_channels=64, kernel_size=3, strides=1, padding="SAME", name="conv", verbose=verbose)
            x = x + self.block1
            with tf.variable_scope("pool1"):
                x = MaxPooling2d(x, name="maxpooling", pool_size=2, strides=2, padding="SAME", verbose=verbose)

            with tf.variable_scope("conv2d_1_2"):
                x = Conv2d(x, out_channels=128, kernel_size=3, strides=1, padding="SAME", name="conv", verbose=verbose)
                x = LeakyReLU(x, name="relu", verbose=verbose)
            self.block2 = x
            with tf.variable_scope("conv2d_3"):
                x = BatchNorm(x, name="batchnorm", training=is_training, verbose=verbose)
                x = LeakyReLU(x, name="relu", verbose=verbose)
                x = Conv2d(x, out_channels=128, kernel_size=3, strides=1, padding="SAME", name="conv", verbose=verbose)
            with tf.variable_scope("conv2d_4"):
                x = BatchNorm(x, name="batchnorm", training=is_training, verbose=verbose)
                x = LeakyReLU(x, name="relu", verbose=verbose)
                x = Conv2d(x, out_channels=128, kernel_size=3, strides=1, padding="SAME", name="conv", verbose=verbose)
            x = x + self.block2
            with tf.variable_scope("pool2"):
                x = MaxPooling2d(x, name="maxpooling", pool_size=2, strides=2, padding="SAME", verbose=verbose)

            with tf.variable_scope("conv2d_2_3"):
                x = Conv2d(x, out_channels=256, kernel_size=3, strides=1, padding="SAME", name="conv", verbose=verbose)
                x = LeakyReLU(x, name="relu", verbose=verbose)
            self.block3 = x
            with tf.variable_scope("conv2d_5"):
                x = BatchNorm(x, name="batchnorm", training=is_training, verbose=verbose)
                x = LeakyReLU(x, name="relu", verbose=verbose)
                x = Conv2d(x, out_channels=256, kernel_size=3, strides=1, padding="SAME", name="conv", verbose=verbose)
            with tf.variable_scope("conv2d_6"):
                x = BatchNorm(x, name="batchnorm", training=is_training, verbose=verbose)
                x = LeakyReLU(x, name="relu", verbose=verbose)
                x = Conv2d(x, out_channels=256, kernel_size=3, strides=1, padding="SAME", name="conv", verbose=verbose)
            x = x + self.block3
            with tf.variable_scope("pool3"):
                x = MaxPooling2d(x, name="maxpooling", pool_size=2, strides=2, padding="SAME", verbose=verbose)

            with tf.variable_scope("conv2d_3_4"):
                x = Conv2d(x, out_channels=512, kernel_size=3, strides=1, padding="SAME", name="conv", verbose=verbose)
                x = LeakyReLU(x, name="relu", verbose=verbose)
            self.block4 = x
            with tf.variable_scope("conv2d_7"):
                x = BatchNorm(x, name="batchnorm", training=is_training, verbose=verbose)
                x = LeakyReLU(x, name="relu", verbose=verbose)
                x = Conv2d(x, out_channels=512, kernel_size=3, strides=1, padding="SAME", name="conv", verbose=verbose)
            with tf.variable_scope("conv2d_8"):
                x = BatchNorm(x, name="batchnorm", training=is_training, verbose=verbose)
                x = LeakyReLU(x, name="relu", verbose=verbose)
                x = Conv2d(x, out_channels=512, kernel_size=3, strides=1, padding="SAME", name="conv", verbose=verbose)
            x = x + self.block4

            ##

        print("+---------------------------------------------------------------------------------------+")
        print("|                                   Model established                                   |")
        print("+---------------------------------------------------------------------------------------+")
        return x