#! /usr/bin/python
# -*- coding: utf8 -*
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# a = np.random.rand(1, 3, 3, 1)
# b = np.random.rand(1, 3, 3, 1)
# c = np.float32(np.mean(np.square(a - b)**2))
# d = np.multiply(10.0, np.log(1.0 * 1.0 / c) / np.log(10.0))
# print(c)
# print(type(c))
# print(d)
# print(type(d))

test_file_list = sorted(glob.glob("data/Test/Set12_BM3D_25" + '/*.png'))
GT_file_list = sorted(glob.glob("data/Test/Set12" + '/*.png'))

test_img = cv2.imread(test_file_list[2], 0) / 255
GT_img = cv2.imread(GT_file_list[2], 0) / 255

plt.figure()
plt.imshow(test_img)
plt.figure()
plt.imshow(GT_img)

test_img = test_img.reshape(1, test_img.shape[0], test_img.shape[1], 1)
GT_img =GT_img.reshape(1, GT_img.shape[0], GT_img.shape[1], 1)

mse = np.float32(np.mean(np.square(test_img - GT_img)))

print(type(mse))
psnr = np.multiply(10.0, np.log(1.0 * 1.0 / mse) / np.log(10.0))
print(psnr)
