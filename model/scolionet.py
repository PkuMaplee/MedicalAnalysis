import numpy as np
import tensorflow as tf
from utils.layer import *


class ScolioNet(object):

    def __init__(self, x, landmarks, is_training, verbose=False):
        self.x = x
        self.landmarks = landmarks
        self.n_landmarks = self.landmarks.get_shape().as_list()[-1]
        self.is_training = is_training

        self.in_block1 = None
        self.in_block2 = None
        self.in_block3 = None
        self.in_block4 = None
        self.var_list = []
        self.midoutputs = []

        self.predictions = self.build_model(x, self.is_training, verbose=verbose)
        self.loss = self.get_loss(self.predictions, self.landmarks)

    def build_model(self, x, is_training, verbose):
        print("+---------------------------------------------------------------------------------------+")
        print("|                            Building the model -- ScolioNet                            |")
        print("+---------------------------------------------------------------------------------------+")
        print("|                                                                                       |")
        self.midoutputs.append(x)
        with tf.variable_scope("scolionet"):
            with tf.variable_scope("conv2d_0_1"):
                x = Conv2d(x, out_channels=64, kernel_size=3, strides=1, padding="SAME", name="conv", verbose=verbose)
            self.in_block1 = x
            self.midoutputs.append(x)
            with tf.variable_scope("residual_block1_1"):
                x = BatchNorm(x, name="batchnorm", training=is_training, verbose=verbose)
                x = LeakyReLU(x, name="leakyrelu", verbose=verbose)
                x = Conv2d(x, out_channels=64, kernel_size=3, strides=1, padding="SAME", name="conv", verbose=verbose)
            with tf.variable_scope("residual_block1_2"):
                x = BatchNorm(x, name="batchnorm", training=is_training, verbose=verbose)
                x = LeakyReLU(x, name="leakyrelu", verbose=verbose)
                x = Conv2d(x, out_channels=64, kernel_size=3, strides=1, padding="SAME", name="conv", verbose=verbose)
            x = x + self.in_block1
            self.midoutputs.append(x)
            with tf.variable_scope("pool1"):
                x = MaxPooling2d(x, name="maxpooling", pool_size=2, strides=2, padding="SAME", verbose=verbose)

            with tf.variable_scope("conv2d_1_2"):
                x = Conv2d(x, out_channels=128, kernel_size=3, strides=1, padding="SAME", name="conv", verbose=verbose)
                x = LeakyReLU(x, name="leakyrelu", verbose=verbose)
            self.in_block2 = x
            self.midoutputs.append(x)
            with tf.variable_scope("residual_block2_1"):
                x = BatchNorm(x, name="batchnorm", training=is_training, verbose=verbose)
                x = LeakyReLU(x, name="leakyrelu", verbose=verbose)
                x = Conv2d(x, out_channels=128, kernel_size=3, strides=1, padding="SAME", name="conv", verbose=verbose)
            with tf.variable_scope("residual_block2_2"):
                x = BatchNorm(x, name="batchnorm", training=is_training, verbose=verbose)
                x = LeakyReLU(x, name="leakyrelu", verbose=verbose)
                x = Conv2d(x, out_channels=128, kernel_size=3, strides=1, padding="SAME", name="conv", verbose=verbose)
            x = x + self.in_block2
            self.midoutputs.append(x)
            with tf.variable_scope("pool2"):
                x = MaxPooling2d(x, name="maxpooling", pool_size=2, strides=2, padding="SAME", verbose=verbose)

            with tf.variable_scope("conv2d_2_3"):
                x = Conv2d(x, out_channels=256, kernel_size=3, strides=1, padding="SAME", name="conv", verbose=verbose)
                x = LeakyReLU(x, name="leakyrelu", verbose=verbose)
            self.in_block3 = x
            self.midoutputs.append(x)
            with tf.variable_scope("residual_block3_1"):
                x = BatchNorm(x, name="batchnorm", training=is_training, verbose=verbose)
                x = LeakyReLU(x, name="leakyrelu", verbose=verbose)
                x = Conv2d(x, out_channels=256, kernel_size=3, strides=1, padding="SAME", name="conv", verbose=verbose)
            with tf.variable_scope("residual_block3_2"):
                x = BatchNorm(x, name="batchnorm", training=is_training, verbose=verbose)
                x = LeakyReLU(x, name="leakyrelu", verbose=verbose)
                x = Conv2d(x, out_channels=256, kernel_size=3, strides=1, padding="SAME", name="conv", verbose=verbose)
            x = x + self.in_block3
            self.midoutputs.append(x)
            with tf.variable_scope("pool3"):
                x = MaxPooling2d(x, name="maxpooling", pool_size=2, strides=2, padding="SAME", verbose=verbose)

            with tf.variable_scope("conv2d_3_4"):
                x = Conv2d(x, out_channels=512, kernel_size=3, strides=1, padding="SAME", name="conv", verbose=verbose)
                x = LeakyReLU(x, name="relu", verbose=verbose)
            self.in_block4 = x
            self.midoutputs.append(x)
            with tf.variable_scope("residual_block4_1"):
                x = BatchNorm(x, name="batchnorm", training=is_training, verbose=verbose)
                x = LeakyReLU(x, name="leakyrelu", verbose=verbose)
                x = Conv2d(x, out_channels=512, kernel_size=3, strides=1, padding="SAME", name="conv", verbose=verbose)
            with tf.variable_scope("residual_block4_2"):
                x = BatchNorm(x, name="batchnorm", training=is_training, verbose=verbose)
                x = LeakyReLU(x, name="leakyrelu", verbose=verbose)
                x = Conv2d(x, out_channels=512, kernel_size=3, strides=1, padding="SAME", name="conv", verbose=verbose)
            x = x + self.in_block4
            self.midoutputs.append(x)
            with tf.variable_scope("pool4"):
                x = MaxPooling2d(x, name="maxpooling", pool_size=2, strides=2, padding="SAME", verbose=verbose)
            with tf.variable_scope("conv2d_4"):
                x = Conv2d(x, out_channels=256, kernel_size=3, strides=1, padding="SAME", name="conv", verbose=verbose)
                x = LeakyReLU(x, name="leakyrelu", verbose=verbose)
            self.midoutputs.append(x)
            # ======== regression components ======== #
            with tf.variable_scope("flatten"):
                x = Flatten(x, name="flatten", verbose=verbose)
            with tf.variable_scope("dense_1"):
                x = Dense(x, units=256, name="dense", verbose=verbose)
                x = LeakyReLU(x, name="leakyrelu", verbose=verbose)
            self.midoutputs.append(x)
            with tf.variable_scope("dense_2"):
                x = Dense(x, units=self.n_landmarks, name="dense", verbose=verbose)
                # x = LeakyReLU(x, name="leakyrelu", verbose=verbose)
                # x = Softmax(x, name="softmax", verbose=verbose)

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="scolionet")
        print("+---------------------------------------------------------------------------------------+")
        print("|                                   Model established                                   |")
        print("+---------------------------------------------------------------------------------------+")
        return x

    def get_loss(self, predictions, landmarks):
        predictions = tf.reshape(predictions, shape=[-1, self.n_landmarks/2, 2])
        landmarks = tf.reshape(landmarks, shape=[-1, self.n_landmarks/2, 2])
        euclidean_dist_diff = tf.sqrt(tf.reduce_sum(tf.squared_difference(predictions, landmarks), 2))
        mean_euclidean_dist_diff = tf.reduce_mean(euclidean_dist_diff, 1)
        loss = tf.reduce_mean(mean_euclidean_dist_diff)
        return loss

