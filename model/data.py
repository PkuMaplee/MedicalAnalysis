#!/usr/bin/env python
# -*- coding: utf-8 -*-


# ==================================================================================================================== #
#                                                         Data                                                         #
# ==================================================================================================================== #
import numpy as np
import glob
import os
import cv2
from numpy import genfromtxt
import random
# from sklearn.model_selection import train_test_split

from utils.utils import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

class DataItem(object):
    
    def __init__(self, index="", folder=""):
        self.index = index
        self.folder = folder
        self.image = None
        self.image_label = None
        self.label = None
        self.floatlabel = None
        self.L1 = None
        self.L1_int = None
        self.S1 = None
        self.S1_int = None
        self.T5 = None
        self.T5_int = None
        self.T12 = None
        self.T12_int = None
        self.fem_center = None
        self.fem_center_int = None
        self.init()

    def init(self):
        if self.index != "":
            self.loadL1(self.index)
            self.loadS1(self.index)
            self.loadT5(self.index)
            self.loadT12(self.index)
            self.loadfem(self.index)
            self.loadimage(self.index)
            self.genelabel()
        else:
            pass
    
    def loadL1(self, index):
        path = os.path.join(self.folder, str(index) + "_lat_lab_L1.txt")
        self.L1 = genfromtxt(path, dtype=np.float32)
        self.L1_int = self.L1.astype(np.int32)
    
    def loadS1(self, index):
        path = os.path.join(self.folder, str(index) + "_lat_lab_S1.txt")
        self.S1 = genfromtxt(path, dtype=np.float32)
        self.S1_int = self.S1.astype(np.int32)
    
    def loadT5(self, index):
        path = os.path.join(self.folder, str(index) + "_lat_lab_T5.txt")
        self.T5 = genfromtxt(path, dtype=np.float32)
        self.T5_int = self.T5.astype(np.int32)
    
    def loadT12(self, index):
        path = os.path.join(self.folder, str(index) + "_lat_lab_T12.txt")
        self.T12 = genfromtxt(path, dtype=np.float32)
        self.T12_int = self.T12.astype(np.int32)
    
    def loadfem(self, index):
        path = os.path.join(self.folder, str(index) + "_lat_lab_fem*.txt")
        femtxtlist = glob.glob(path)
        self.fem_center = np.zeros([2], dtype=np.float32)
        for i in range(len(femtxtlist)):
            arc = genfromtxt(femtxtlist[i], dtype=np.float32)
            self.fem_center += np.mean(arc, axis=0)
        self.fem_center /= len(femtxtlist)
        self.fem_center_int = self.fem_center.astype(np.int32)

    def loadimage(self, index):
        self.image = cv2.imread(os.path.join(self.folder, str(index) + "_lat.png"), 0)
        self.image_label = cv2.imread(os.path.join(self.folder, str(index) + "_lat_lab.png"))

    def genelabel(self):
        self.floatlabel = np.array([], dtype=np.float32)
        self.floatlabel = np.append(self.floatlabel, self.T5.reshape(-1))
        self.floatlabel = np.append(self.floatlabel, self.T12.reshape(-1))
        self.floatlabel = np.append(self.floatlabel, self.L1.reshape(-1))
        self.floatlabel = np.append(self.floatlabel, self.S1.reshape(-1))
        self.floatlabel = np.append(self.floatlabel, self.fem_center.reshape(-1))
        self.label = self.floatlabel.astype(np.int32)

    def update(self):
        self.T5_int = self.T5.astype(np.int32)
        self.T12_int = self.T12.astype(np.int32)
        self.L1_int = self.L1.astype(np.int32)
        self.S1_int = self.S1.astype(np.int32)
        self.fem_center_int = self.fem_center.astype(np.int32)
        self.floatlabel = np.array([], dtype=np.float32)
        self.floatlabel = np.append(self.floatlabel, self.T5.reshape(-1))
        self.floatlabel = np.append(self.floatlabel, self.T12.reshape(-1))
        self.floatlabel = np.append(self.floatlabel, self.L1.reshape(-1))
        self.floatlabel = np.append(self.floatlabel, self.S1.reshape(-1))
        self.floatlabel = np.append(self.floatlabel, self.fem_center.reshape(-1))
        self.label = self.floatlabel.astype(np.int32)

    def getlabel(self):
        return self.label

    def vis_check(self, save=False):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(121)
        ax.imshow(self.image, cmap="gray")
        lineT5 = Line2D(self.T5_int[:, 0], self.T5_int[:, 1], c="red", marker="o", markersize=3, label="T5")
        lineT12 = Line2D(self.T12_int[:, 0], self.T12_int[:, 1], c="steelblue", marker="o", markersize=3, label="T12")
        lineL1 = Line2D(self.L1_int[:, 0], self.L1_int[:, 1], c="orange", marker="o", markersize=3, label="L1")
        lineS1 = Line2D(self.S1_int[:, 0], self.S1_int[:, 1], c="purple", marker="o", markersize=3, label="S1")
        ax.add_line(lineT5)
        ax.add_line(lineT12)
        ax.add_line(lineL1)
        ax.add_line(lineS1)
        ax.scatter(self.fem_center[0], self.fem_center[1], c="green", marker="X", s=60, label="femoral")
        plt.title("Check With Labels")
        ax.legend()

        ax = fig.add_subplot(122)
        ax.imshow(self.image_label, cmap="gray")
        plt.title("Image With Labels")

        if save:
            if not os.path.exists("data/check"):
                os.makedirs("data/check")
            plt.savefig(os.path.join("data/check", str(self.index) + "_check.jpg"),
                        interpolation="nearest",
                        transparent=True,
                        bbox_inches="tight")



class Data(object):

    def __init__(self, path="", configs=None, train_test_split_rate=0.3, random_seed=36, savepath="augment"):
        self.path = path
        self.configs = configs
        self.savepath = savepath
        self.rate = train_test_split_rate
        self.random_seed = random_seed
        self.data = []
        self.traindata = []
        self.testdata = []
        self.num_samples = None
        # check if path/to/data/data_clean is founded, if not run:
        self.readraw()
        self.cleaning()
        self.train_test_split()
        self.savedata()
        # if found run:
        # self.readcleandata()
    
    def readraw(self):
        self.imglist = glob.glob(os.path.join(self.path, "*lat.png"))
        self.imglist.sort()
        indices = [item.split("/")[-1].split("_")[0] for item in self.imglist]

        for i in range(len(indices)):
            Item = DataItem(index=indices[i], folder=self.path)
            self.data.append(Item)

        self.num_samples = len(self.data)

    
    def train_test_split(self):
        random.seed(self.random_seed)
        random.shuffle(self.data)

        self.testdata = self.data[:np.int(self.rate * self.num_samples)]
        self.traindata = self.data[np.int(self.rate * self.num_samples):]

        self.num_train = len(self.traindata)
        self.num_test = len(self.testdata)
        return
    
    def cleaning(self):
        '''
        Clean the training data:
        [1]. Transform the training image to the consistant size (crop and resize).
        :return: 
        '''
        height = self.configs["height"]
        width = self.configs["width"]
        for i in range(self.num_samples):
            item = self.data[i]
            img_h, img_w = item.image.shape[:2]
            if img_w > width:
                item = horizon_centercrop(item, width)
            if img_h > height:
                item = vertical_topcrop(item, height)
            item = standard_resize(item, [height, width])
            item.vis_check(save=True)
            self.data[i] = item

    
    def savedata(self):
        '''
        Store the augmented train data and test data to a local repository
        :return:
        '''
        pass
