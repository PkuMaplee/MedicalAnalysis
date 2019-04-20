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
#                                                    MaxPooling                                                        #
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