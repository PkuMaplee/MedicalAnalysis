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

class DataItem(object):
    
    def __init__(self, index="", folder=""):
        self.index = index
        self.folder = folder
        self.image = None
        self.image_label = None
        self.label = None
        self.L1 = None
        self.S1 = None
        self.T5 = None
        self.T12 = None
        self.fem_center = None
        self.init()

    def init(self):
        if self.index != "":
            self.loadL1(self.index)
            self.loadS1(self.index)
            self.loadT5(self.index)
            self.loadT12(self.index)
            self.loadfem(self.index)
            self.loadimage(self.index)
            self.getlabel()
        else:
            pass
    
    def loadL1(self, index):
        path = os.path.join(self.folder, str(index) + "_lat_lab_L1.txt")
        self.L1 = genfromtxt(path, dtype=np.float32)
    
    def loadS1(self, index):
        path = os.path.join(self.folder, str(index) + "_lat_lab_S1.txt")
        self.S1 = genfromtxt(path, dtype=np.float32)
    
    def loadT5(self, index):
        path = os.path.join(self.folder, str(index) + "_lat_lab_T5.txt")
        self.T5 = genfromtxt(path, dtype=np.float32)
    
    def loadT12(self, index):
        path = os.path.join(self.folder, str(index) + "_lat_lab_T12.txt")
        self.T12 = genfromtxt(path, dtype=np.float32)
    
    def loadfem(self, index):
        path = os.path.join(self.folder, str(index) + "_lat_lab_fem*.txt")
        femtxtlist = glob.glob(path)
        self.fem_center = np.zeros([2], dtype=np.float32)
        for i in range(len(femtxtlist)):
            arc = genfromtxt(femtxtlist[i], dtype=np.float32)
            self.fem_center += np.mean(arc, axis=0)
        self.fem_center /= len(femtxtlist)
        return self.fem_center

    def loadimage(self, index):
        self.image = cv2.imread(os.path.join(self.folder, str(index) + "_lat.png"), 0)
        self.image_label = cv2.imread(os.path.join(self.folder, str(index) + "_lat_lab.png"), 0)

    def getlabel(self):
        self.label = np.array([], dtype=np.float32)
        self.label = np.append(self.label, self.T5.reshape(-1))
        self.label = np.append(self.label, self.T12.reshape(-1))
        self.label = np.append(self.label, self.L1.reshape(-1))
        self.label = np.append(self.label, self.S1.reshape(-1))
        self.label = np.append(self.label, self.fem_center.reshape(-1))



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
        self.readraw()
        self.train_test_split()
        self.cleaning()
        self.savedata()
    
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
    
    def clearning(self):
        '''
        Clean the training data:
        [1]. Transform the training image to the consistant size (crop and resize).
        :return: 
        '''



    
    def savedata(self):
        '''
        Store the augmented train data and test data to a local repository
        :return:
        '''
        pass
