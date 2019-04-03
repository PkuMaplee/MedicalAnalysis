#!/usr/bin/env python
# -*- coding: utf-8 -*-


# -------------------------------------------------------------------------------------------------------------------- #
#   Description:                                                                                                       #
#       This python script is the main file for the entire project, which starts to train and test (validate)          #
#                                                                                                                      #
#   Contact:                                                                                                           #
#       Nan Meng                                                                                                       #
#       naen.mong@gmail.com                                                                                            #
#       University of Hong Kong                                                                                        #
# -------------------------------------------------------------------------------------------------------------------- #

from __future__ import division, print_function, absolute_import
from model.data import *

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="data/Lat1-100", help="Path to the dataset")
parser.add_argument("--model_type", default="default", help="Choose the model")
parser.add_argument("--channels", type=int, default=1, help="The number of input image channels")
parser.add_argument("--batchsize", type=int, default=128, help="The number of input images in each batch")
parser.add_argument("--height", type=int, default=860, help="The size of input images")
parser.add_argument("--width", type=int, default=360, help="The size of input images")
parser.add_argument("--learning_rate", type=float, default=0.000002, help="The learning rate for training")
parser.add_argument("--learning_rate_step", type=int, default=50)
parser.add_argument("--num_epoch", type=int, default=100)
parser.add_argument("--display_step", type=int, default=1)
parser.add_argument("--save_step", type=int, default=5)
parser.add_argument("--select_gpu", default='3')
parser.add_argument("--results_folder", default="results")
parser.add_argument("--weights_folder", default="weights")

args = parser.parse_args()



def main(args):

    # ====== Load configurations ====== #
    configs = {
        "path": args.dataset,
        "batchSize": args.batchsize,
        "height": args.height,
        "width": args.width,
        "channels": args.channels,
        "display_step": args.display_step,
        "learning_rate": args.learning_rate,
        "lr_step": args.learning_rate_step,
        "select_gpu": args.select_gpu,
        "results_folder": args.results_folder,
        "weights_folder": args.weights_folder
    }

    data = Data("data/Lat1-100", configs=configs)
    # ====== Model definition ====== #
    # model = Classifier(args.model_type, configs=configs)

    # ====== Start training ====== #
    # model.train(num_epoch=args.num_epoch, save_epoch=args.save_step, continues=True)

    # ====== Start testing ====== #
    # model.valid()
    # model.test()

    print(0)


if __name__ == "__main__":
    main(args)