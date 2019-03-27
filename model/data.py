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
from sklearn.model_selection import train_test_split


class Data(object):

    def __init__(self, path="", num_labels=18, train_test_split_rate=0.3, savepath=""):
        self.path = path
        self.savepath = savepath
        self.num_labels = num_labels
        self.rate = train_test_split_rate
        self.data = None
        self.traindata = None
        self.testdata = None
        # self.validdata = None
        self.num_samples = None
        self.readraw()
        self.train_test_split()
        self.augment()
        self.savedata()
    
    def readraw(self):
        self.imglist = glob.glob(os.path.join(self.path, "*lat.png"))
        self.imglist.sort()
        self.imglablist = []
        self.num_samples = len(self.imglist)
        self.labels = np.zeros([self.num_samples, self.num_labels], dtype=np.float32)
        for i in range(self.num_samples):
            imgname = self.imglist[i]
            prefix = imgname.split("/")[-1].split(".")[0]
            tmppath = os.path.join(self.path, prefix + "_lab.png")
            self.imglablist.append(tmppath)
            tmppath = os.path.join(self.path, prefix + "_lab_L1.txt")
            l1 = self.readline(tmppath)
            self.labels[i, 0:4] = l1
            tmppath = os.path.join(self.path, prefix + "_lab_S1.txt")
            l2 = self.readline(tmppath)
            self.labels[i, 4:8] = l2
            tmppath = os.path.join(self.path, prefix + "_lab_T5.txt")
            l3 = self.readline(tmppath)
            self.labels[i, 8:12] = l3
            tmppath = os.path.join(self.path, prefix + "_lab_T12.txt")
            l4 = self.readline(tmppath)
            self.labels[i, 12:16] = l4
            tmppath = os.path.join(self.path, prefix + "_lab_fem*.txt")
            c1 = self.readcenter(tmppath)
            self.labels[i, 16:] = c1
        self.data = {"images": self.imglist, "image_labels": self.imglablist, "labels": self.labels}
        return

    def readline(self, path):
        line = genfromtxt(path, dtype=np.float32)
        line = line.reshape(-1)
        return line

    def readcenter(self, path):
        filelist = glob.glob(path)
        center = np.zeros([2], dtype=np.float32)
        for i in range(len(filelist)):
            arc = genfromtxt(filelist[i], dtype=np.float32)
            center += np.mean(arc, axis=0)
        center /= len(filelist)
        return center
    
    def train_test_split(self):
        trainimgs, testimgs, trainimglabels, testimglabels = train_test_split(self.data["images"], 
                                                                              self.data["image_labels"], 
                                                                              test_size=self.rate, 
                                                                              random_state=36)
        trainimgs, testimgs, trainlabels, testlabels = train_test_split(self.data["images"], 
                                                                        self.data["labels"], 
                                                                        test_size=self.rate, 
                                                                        random_state=36)
        self.traindata = {"images": trainimgs, "image_labels": trainimglabels, "labels": trainlabels}
        self.testdata = {"images": testimgs, "image_labels": testimglabels, "labels": testlabels}
        return
    
    def augment(self):
        '''
        augment the training data
        :return: 
        '''
        
        return
    
    def savedata(self):
        return
