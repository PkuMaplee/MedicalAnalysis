#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cv2
import re
import tensorflow as tf
import tensorflow_probability as tfp
from utils.utils import *
from model.data import *
from model.scolionet import *
import matplotlib.pyplot as plt

from tool.log_config import *


# log_config()
# tf.logging.set_verbosity(tf.logging.INFO)


class Detector(object):

    def __init__(self, data, model_type="", configs=None, verbose=True):
        self.data = data
        self.configs = configs
        self.model_type = model_type.lower()
        self.model = None
        # ===================== get settings ===================== #
        self.weights_folder = self.configs["weights_folder"]
        self.display_step = self.configs["display_step"]
        self.lr_step = self.configs["lr_step"]
        self.results_folder = self.configs["results_folder"]

        # ==================== initialization ==================== #
        self.is_training = tf.placeholder(tf.bool, [])
        self.lr = tf.placeholder(tf.float32, [])
        self.inputs = tf.placeholder(tf.float32, [None, self.configs["imageHeight"],
                                                  self.configs["imageWidth"],
                                                  self.configs["channels"]])
        self.landmarks = tf.placeholder(tf.float32, [None, self.configs["num_landmarks"] * 2])

        # =================== Building the model ================= #
        self.build_model(verbose)

    def build_model(self, verbose):
        if self.model_type == "scolionet":
            self.model = ScolioNet(self.inputs, self.landmarks, is_training=self.is_training, verbose=verbose)























