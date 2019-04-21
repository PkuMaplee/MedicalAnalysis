#!/usr/bin/env python
# -*- coding: utf-8 -*-


# -------------------------------------------------------------------------------------------------------------------- #
#
#
#
#
# -------------------------------------------------------------------------------------------------------------------- #

import tensorflow as tf
import numpy as np


def Conv2d(x, out_channels, kernel_size=3, strides=1, padding="SAME", trainable=True, name=None, verbose=False):

    out = tf.layers.conv2d(inputs=x,
                           filters=out_channels,
                           kernel_size=kernel_size,
                           strides=strides,
                           padding=padding,
                           kernel_initializer=tf.glorot_uniform_initializer(),
                           trainable=trainable,
                           name=name)
    if verbose:
        print("|------------------------------------ Convolution2D ------------------------------------|")
        print("| feature size: {0: <27}  filter size: {1: <30}|".format(out.shape,
                                                                        (kernel_size, kernel_size,
                                                                         x.shape.as_list()[-1], out_channels)))

    return out


# ==================================================================================================================== #
#                                              Activation Functions                                                    #
# ==================================================================================================================== #


def ReLU(x, name=None, verbose=False):

    out = tf.nn.relu(features=x, name=name)
    if verbose:
        print("|----------------------------------- ReLU Activation -----------------------------------|")
        print("| feature size: {0: <72}|".format(out.shape))

    return out


def LeakyReLU(x, name=None, alpha=0.2, verbose=False):

    out = tf.nn.leaky_relu(features=x, alpha=alpha, name=name)
    if verbose:
        print("|--------------------------------- LeakyReLU Activation --------------------------------|")
        print("| feature size: {0: <72}|".format(out.shape))

    return out

# ==================================================================================================================== #
#                                               Batch-Normalization                                                    #
# ==================================================================================================================== #


def BatchNorm(x, name=None, axis=-1, momentum=0.99, epsilon=0.001, training=True, reuse=None, verbose=False):

    out = tf.layers.batch_normalization(inputs=x,
                                        axis=axis,
                                        momentum=momentum,
                                        epsilon=epsilon,
                                        name=name,
                                        training=training,
                                        reuse=reuse)
    if verbose:
        print("|--------------------------------- Batch Normalization ---------------------------------|")
        print("| feature size: {0: <72}|".format(out.shape))

    return out

# ==================================================================================================================== #
#                                                      Pooling                                                         #
# ==================================================================================================================== #


def MaxPooling2d(x, name=None, pool_size=2, strides=1, padding="SAME", verbose=False):

    out = tf.layers.max_pooling2d(inputs=x,
                                  pool_size=pool_size,
                                  strides=strides,
                                  padding=padding,
                                  name=name)
    if verbose:
        print("|------------------------------------- MaxPooling --------------------------------------|")
        print("| feature size: {0: <72}|".format(out.shape))

    return out


# ==================================================================================================================== #
#                                                 Other Functions                                                      #
# ==================================================================================================================== #


def PixelShuffle(x, r, n_split, verbose=False):

    def PS(x, r):
        bs, a, b, c = x.get_shape().as_list()
        x = tf.reshape(x, (bs, a, b, r, r))
        x = tf.transpose(x, (0, 1, 2, 4, 3))
        x = tf.split(x, a, 1)
        x = tf.concat([tf.squeeze(x_) for x_ in x], 2)
        x = tf.split(x, b, 1)
        x = tf.concat([tf.squeeze(x_) for x_ in x], 2)
        return tf.reshape(x, (bs, a * r, b * r, 1))

    xc = tf.split(x, n_split, 3)

    out = tf.concat([PS(x_, r) for x_ in xc], 3)

    if verbose:
        print("|------------------------------------ PixelShuffle -------------------------------------|")
        print("| feature size: {0: <72}|".format(out.shape))

    return out


def Deconv2d(x, out_channels, kernel_size=3, strides=1, padding="SAME", trainable=True, name=None, verbose=False):

    out = tf.layers.conv2d_transpose(inputs=x,
                                     filters=out_channels,
                                     kernel_size=kernel_size,
                                     strides=strides,
                                     padding=padding,
                                     kernel_initializer=tf.glorot_uniform_initializer(),
                                     trainable=trainable,
                                     name=name)
    if verbose:
        print("|------------------------------- Convolution2D Transpose -------------------------------|")
        print("| feature size: {0: <72}|".format(out.shape))

    return out


def Flatten(x, name=None, verbose=False):

    out = tf.layers.flatten(inputs=x, name=name)
    if verbose:
        print("|--------------------------------------- Flatten ---------------------------------------|")
        print("| feature size: {0: <72}|".format(out.shape))

    return out


def Dense(x, units, name=None, verbose=False):

    out = tf.layers.dense(inputs=x,
                          units=units,
                          kernel_initializer=tf.glorot_uniform_initializer(),
                          name=name)
    if verbose:
        print("|---------------------------------------- Dense ----------------------------------------|")
        print("| feature size: {0: <72}|".format(out.shape))

    return out


def Softmax(x, axis=-1, name=None, verbose=False):

    out = tf.nn.softmax(logits=x, axis=axis, name=name)

    if verbose:
        print("|--------------------------------------- Softmax ---------------------------------------|")
        print("| feature size: {0: <72}|".format(out.shape))

    return out