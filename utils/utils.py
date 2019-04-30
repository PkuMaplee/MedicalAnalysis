import numpy as np
import cv2
import matplotlib.pyplot as plt

# def augment(images, labels):
#     return


def preresize(item, standard):
    standh, standw = standard
    ori_h, ori_w = item.image.shape[:2]
    y_scale = np.float32(standh+100) / np.float(ori_h)
    x_scale = np.float32(standw+60) / np.float(ori_w)
    item.image = cv2.resize(item.image, (standw+60, standh+100))
    item.image_label = cv2.resize(item.image_label, (standw+60, standh+100))
    item.T5 *= [x_scale, y_scale]
    item.T12 *= [x_scale, y_scale]
    item.L1 *= [x_scale, y_scale]
    item.S1 *= [x_scale, y_scale]
    item.fem_center *= [x_scale, y_scale]
    item.update()
    return item


def horizon_centercrop(item, width):
    h, w = item.image.shape[:2]
    if w > width:
        start_w = (w - width) / 2
        end_w = start_w + width
        item.image = item.image[:, start_w:end_w, ...]
        item.image_label = item.image_label[:, start_w:end_w, ...]
        item.T5[:, 0] = item.T5[:, 0] - start_w
        item.T12[:, 0] = item.T12[:, 0] - start_w
        item.L1[:, 0] = item.L1[:, 0] - start_w
        item.S1[:, 0] = item.S1[:, 0] - start_w
        item.fem_center[0] = item.fem_center[0] - start_w
        item.update()
    return item


def vertical_topcrop(item, height):
    h, w = item.image.shape[:2]
    if h > height:
        start_h = h - height
        item.image = item.image[start_h:, ...]
        item.image_label = item.image_label[start_h:, ...]
        item.T5[:, 1] = item.T5[:, 1] - start_h
        item.T12[:, 1] = item.T12[:, 1] - start_h
        item.L1[:, 1] = item.L1[:, 1] - start_h
        item.S1[:, 1] = item.S1[:, 1] - start_h
        item.fem_center[1] = item.fem_center[1] - start_h
        item.update()
    return item


def standard_resize(item, standard):
    standh, standw = standard
    ori_h, ori_w = item.image.shape[:2]
    y_scale = np.float32(standh) / np.float(ori_h)
    x_scale = np.float32(standw) / np.float(ori_w)
    item.image = cv2.resize(item.image, (standw, standh))
    item.image_label = cv2.resize(item.image_label, (standw, standh))
    item.T5 *= [x_scale, y_scale]
    item.T12 *= [x_scale, y_scale]
    item.L1 *= [x_scale, y_scale]
    item.S1 *= [x_scale, y_scale]
    item.fem_center *= [x_scale, y_scale]
    item.update()
    return item


def denormalize_label():
    pass
