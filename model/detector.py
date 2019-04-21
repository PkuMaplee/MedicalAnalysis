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

        self.sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.saver = tf.train.Saver()
        self.opt = None

        # =================== Building the model ================= #
        self.build_model(verbose)

    def build_model(self, verbose):
        if self.model_type == "scolionet":
            self.model = ScolioNet(self.inputs, self.landmarks, is_training=self.is_training, verbose=verbose)

    def adjust_learning_rate(self, lr, epoch, step=10):
        return lr * (0.1 ** (epoch // step))

    def train(self, num_epoch=100, opt="adam", save_epoch=5, test_epoch=1, continues=False, model_path="latest.ckpt"):
        if opt == "adam":
            self.opt = tf.train.AdamOptimizer(beta1=0.9, learning_rate=self.lr)

        train_op = self.opt.minimize(self.model.loss, global_step=self.global_step, var_list=self.model.var_list)
        lr_start = self.configs["learning_rate"]
        lr = lr_start  # self.adjust_learning_rate(lr_start)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        epoch_start = 0
        # if continues:
        #     self.saver.restore(self.sess, os.path.join(self.weights_folder, os.path.join(self.model_type, model_path)))
        #     _, epoch_start = self.read_state()
        #     epoch_start += 1

        for epoch in range(epoch_start, num_epoch):
            random.shuffle(self.data.traindata)
            for iteration in range(30):

                batchimgs, batchlabels = self.data.get_train_batch(iteration=iteration)
                _, loss, predictions = self.sess.run([train_op, self.model.loss, self.model.predictions],
                                                     feed_dict={self.inputs: batchimgs,
                                                                self.landmarks: batchlabels,
                                                                self.is_training: True,
                                                                self.lr: lr})
                string = "[Epoch {:05d}][Iteration {:05d}][TRAIN]\tloss: {:.6f}".format(epoch, iteration, loss)
                print(string)

            for iteration in range(13):
                batchimgs, batchlabels = self.data.get_train_batch(iteration=iteration)
                loss, predictions = self.sess.run([self.model.loss, self.model.predictions],
                                                  feed_dict={self.inputs: batchimgs,
                                                             self.landmarks: batchlabels,
                                                             self.is_training: True,
                                                             self.lr: lr})
                string = "[Epoch {:05d}][Iteration {:05d}][TEST]\tloss: {:.6f}".format(epoch, iteration, loss)
                print(string)






















