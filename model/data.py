#!/usr/bin/env python
# -*- coding: utf-8 -*-


# ==================================================================================================================== #
#                                                      Data Classes                                                    #
# ==================================================================================================================== #

import glob
import os
from numpy import genfromtxt
import random
from tqdm import tqdm
import time
import yaml

from utils.utils import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import warnings
warnings.filterwarnings("ignore")

# ==================================================================================================================== #
#                                                       DataItem                                                       #
# ==================================================================================================================== #

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

    def vis_check(self, save=False, savepath="data/check", show=False):
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
        ax.scatter(self.fem_center_int[0], self.fem_center_int[1], c="green", marker="X", s=60, label="femoral")
        plt.title("Check With Labels")
        ax.legend()

        ax = fig.add_subplot(122)
        ax.imshow(self.image_label, cmap="gray")
        plt.title("Image With Labels")

        if save:
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            plt.savefig(os.path.join(savepath, str(self.index) + "_check.jpg"),
                        interpolation="nearest",
                        transparent=True,
                        bbox_inches="tight")
        if not show:
            plt.close()


    def readyaml(self, path):
        with open(path, "r") as stream:
            try:
                yamldata = yaml.load(stream)
                self.T5 = yamldata["T5"]
                self.T5_int = yamldata["T5_int"]
                self.T12 = yamldata["T12"]
                self.T12_int = yamldata["T12_int"]
                self.L1 = yamldata["L1"]
                self.L1_int = yamldata["L1_int"]
                self.S1 = yamldata["S1"]
                self.S1_int = yamldata["S1_int"]
                self.fem_center = yamldata["fem_center"]
                self.fem_center_int = yamldata["fem_center_int"]
                self.genelabel()
                self.index = path.split("/")[-2]
                self.folder = "/".join(path.split("/")[:3])
                self.image = cv2.imread(yamldata["image"], 0)
                assert self.image is not None, "Can Not Find Image : {}".format(yamldata["image"])
                self.image_label = cv2.imread(yamldata["image_label"])
                assert self.image_label is not None, "Can Not Find Image Label: {}".format(yamldata["image_label"])

            except yaml.YAMLError as exc:
                print(exc)

    def augment(self):
        return


# ==================================================================================================================== #
#                                                         Data                                                         #
# ==================================================================================================================== #

class Data(object):

    def __init__(self, configs=None, train_test_split_rate=0.3, random_seed=36, savepath="augment"):
        self.configs = configs
        self.path = configs["path"]
        self.batchsize = configs["batchSize"]
        self.savepath = savepath
        self.rate = train_test_split_rate
        self.random_seed = random_seed
        self.data = []
        self.traindata = []
        self.testdata = []
        self.num_samples = None
        if not os.path.exists(self.path + "_processed"):
            self.readraw()
            self.cleaning()
            self.train_test_split()
            self.savedata()
        else:
            self.readprocessed()
    
    def readraw(self):
        self.imglist = glob.glob(os.path.join(self.path, "*lat.png"))
        self.imglist.sort()
        indices = [item.split("/")[-1].split("_")[0] for item in self.imglist]

        for i in range(len(indices)):
            Item = DataItem(index=indices[i], folder=self.path)
            self.data.append(Item)

        self.num_samples = len(self.data)

    def readprocessed(self):
        trainlist = glob.glob(os.path.join(self.path + "_processed", "train", "*", "label.yaml"))
        testlist = glob.glob(os.path.join(self.path + "_processed", "test", "*", "label.yaml"))
        self.num_train = len(trainlist)
        self.num_test = len(testlist)
        for i in range(self.num_train):
            item = DataItem()
            item.readyaml(path=trainlist[i])
            self.traindata.append(item)

        for i in range(self.num_test):
            item = DataItem()
            item.readyaml(path=testlist[i])
            self.testdata.append(item)

        self.num_samples = self.num_train + self.num_test
        self.data = self.traindata + self.testdata
    
    def train_test_split(self):
        random.seed(self.random_seed)
        random.shuffle(self.data)

        self.testdata = self.data[:np.int(self.rate * self.num_samples)]
        self.traindata = self.data[np.int(self.rate * self.num_samples):]

        self.num_train = len(self.traindata)
        self.num_test = len(self.testdata)

    def shuffle(self):
        random.shuffle(self.traindata)
    
    def cleaning(self):
        '''
        Clean the training data:
        [1]. Transform the training image to the consistant size (crop and resize).
        :return: 
        '''
        height = self.configs["imageHeight"]
        width = self.configs["imageWidth"]
        for i in range(self.num_samples):
            item = self.data[i]
            img_h, img_w = item.image.shape[:2]
            if img_w > width:
                item = horizon_centercrop(item, width)
            if img_h > height:
                item = vertical_topcrop(item, height)
            item = standard_resize(item, [height, width])
            # item.vis_check(save=True)
            self.data[i] = item

    
    def savedata(self):
        '''
        Store the augmented train data and test data to a local repository
        :return:
        '''
        # if not os.path.exists(os.path.join(self.path + "_processed", "train")):
        #     os.makedirs(os.path.join(self.path + "_processed", "train"))
        # if not os.path.exists(os.path.join(self.path + "_processed", "test")):
        #     os.makedirs(os.path.join(self.path + "_processed", "test"))
        print("Processing the Train Data ......")
        for i in tqdm(range(self.num_train)):
            item = self.traindata[i]
            savefolder = os.path.join(self.path + "_processed", "train", str(item.index))
            if not os.path.exists(savefolder):
                os.makedirs(savefolder)
            img = item.image
            img_label = item.image_label
            cv2.imwrite(os.path.join(savefolder, str(item.index)+".jpg"), img)
            cv2.imwrite(os.path.join(savefolder, str(item.index) + "_label.jpg"), img_label)
            item.vis_check(save=True, savepath=savefolder, show=False)

            data = {"T5": item.T5, "T12": item.T12, "L1": item.L1, "S1": item.S1, "fem_center": item.fem_center,
                    "T5_int": item.T5_int, "T12_int": item.T12_int, "L1_int": item.L1_int, "S1_int": item.S1_int,
                    "fem_center_int": item.fem_center_int, "image": os.path.join(savefolder, str(item.index)+".jpg"),
                    "image_label": os.path.join(savefolder, str(item.index) + "_label.jpg")}

            with open(os.path.join(savefolder, "label.yaml"), "w") as outfile:
                yaml.dump(data, outfile, default_flow_style=False)

        time.sleep(1)
        print("Processing the Test Data ......")
        for i in tqdm(range(self.num_test)):
            item = self.testdata[i]
            savefolder = os.path.join(self.path + "_processed", "test", str(item.index))
            if not os.path.exists(savefolder):
                os.makedirs(savefolder)
            img = item.image
            img_label = item.image_label
            cv2.imwrite(os.path.join(savefolder, str(item.index)+".jpg"), img)
            cv2.imwrite(os.path.join(savefolder, str(item.index) + "_label.jpg"), img_label)
            item.vis_check(save=True, savepath=savefolder, show=False)

            data = {"T5": item.T5, "T12": item.T12, "L1": item.L1, "S1": item.S1, "fem_center": item.fem_center,
                    "T5_int": item.T5_int, "T12_int": item.T12_int, "L1_int": item.L1_int, "S1_int": item.S1_int,
                    "fem_center_int": item.fem_center_int, "image": os.path.join(savefolder, str(item.index)+".jpg"),
                    "image_label": os.path.join(savefolder, str(item.index) + "_label.jpg")}

            with open(os.path.join(savefolder, "label.yaml"), "w") as outfile:
                yaml.dump(data, outfile, default_flow_style=False)

    def normalize_label(self, label):
        label = label.reshape(-1, self.configs["num_landmarks"], 2)
        label = label / [self.configs["imageWidth"], self.configs["imageHeight"]]
        label = label.reshape(-1)
        return label

    def unnormalize_label(self, normlabel):
        normlabel = normlabel.reshape(-1, self.configs["num_landmarks"], 2)
        normlabel = normlabel * [self.configs["imageWidth"], self.configs["imageHeight"]]
        normlabel = normlabel.reshape(-1)
        return normlabel

    def load_batch(self, indices, mode="train"):
        b = self.batchsize
        h = self.configs["imageHeight"]
        w = self.configs["imageWidth"]
        c = self.configs["channels"]

        batchimgs = np.zeros([b, h, w, c], dtype=np.float32)
        batchlabels = np.zeros([b, self.configs["num_landmarks"] * 2], dtype=np.float32)

        if mode.lower() == "train":
            data = self.traindata
        elif mode.lower() == "test":
            data = self.testdata
        else:
            pass
        for i in range(len(indices)):
            image = data[indices[i]].image.astype(np.float32)
            image = image / image.max()
            batchimgs[i] = np.expand_dims(image, axis=-1)
            label = data[indices[i]].label.astype(np.float32)
            # label = self.normalize_label(label)
            batchlabels[i] = label

        return batchimgs, batchlabels

    def next_batch(self):
        pass

    def get_train_batch(self, iteration=0):
        start = self.batchsize * iteration % self.num_train
        end = self.batchsize * (iteration + 1) % self.num_train
        indices = range(self.num_train)
        if start >= end:
            selects = indices[start:] + indices[:end]
        else:
            selects = indices[start:end]

        batch_images, batch_labels = self.load_batch(indices=selects, mode="train")
        return batch_images, batch_labels

    def get_test_batch(self, iteration=0):
        start = self.batchsize * iteration % self.num_test
        end = self.batchsize * (iteration + 1) % self.num_test
        indices = range(self.num_test)
        if start >= end:
            selects = indices[start:] + indices[:end]
        else:
            selects = indices[start:end]

        batch_images, batch_labels = self.load_batch(indices=selects, mode="test")
        return batch_images, batch_labels
