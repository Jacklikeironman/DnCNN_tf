
import glob
import cv2
import numpy as np
import scipy.misc
import scipy.ndimage
import matplotlib.pyplot as plt

patch_size, stride = 40, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128

def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()

def data_aug(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:  # 上下翻转
        return np.flipud(img)
    elif mode == 2:  # 图片逆时针旋转90°
        return np.rot90(img)
    elif mode == 3:  # 对逆时针旋转90度后的图片上下翻转
        return np.flipud(np.rot90(img))
    elif mode == 4:  # 逆时针旋转180度
        return np.rot90(img, k=2)
    elif mode == 5:  # 对逆时针旋转180度后的图片上下翻转
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:  # 逆时针旋转270度
        return np.rot90(img, k=3)
    elif mode == 7:  # 对逆时针旋转270度后的图片上下翻转
        return np.flipud(np.rot90(img, k=3))

def datagenerator(train_data_dir='data/Train400-25', GT_data_dir='data/Train400'):
    train_file_list = sorted(glob.glob(train_data_dir + '/*.png'))  # get name list of all .png files
    GT_file_list = sorted(glob.glob(GT_data_dir + '/*.png'))
    # initialize
    train_data = []
    GT_data = []
    # generate patches
    for i in range(len(train_file_list)):
        train_img = cv2.imread(train_file_list[i], 0)
        GT_img = cv2.imread(GT_file_list[i], 0)
        h, w =train_img.shape
        train_patches = []
        GT_patches = []
        for s in scales:
            h_scaled, w_scaled = int(h * s), int(w * s)
            train_img_scaled = cv2.resize(train_img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
            GT_img_scaled = cv2.resize(GT_img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
            for i in range(0, h_scaled - patch_size + 1, stride):
                for j in range(0, w_scaled - patch_size + 1, stride):
                    train_patch = train_img_scaled[i:i + patch_size, j:j + patch_size]
                    GT_patch = GT_img_scaled[i:i + patch_size, j:j + patch_size]
                    for k in range(0, aug_times):
                        mode = np.random.randint(0, 8)
                        train_patch_aug = data_aug(train_patch, mode)
                        GT_patch_aug = data_aug(GT_patch, mode)
                        train_patches.append(train_patch_aug)
                        GT_patches.append(GT_patch_aug)
        train_data.append(train_patches)
        GT_data.append(GT_patches)
    train_data = np.array(train_data, dtype='uint8')
    GT_data = np.array(GT_data, dtype='uint8')
    train_data = train_data.reshape((train_data.shape[0] * train_data.shape[1], train_data.shape[2], train_data.shape[3], 1))
    GT_data = GT_data.reshape((GT_data.shape[0] * GT_data.shape[1], GT_data.shape[2], GT_data.shape[3], 1))
    discard_n = len(train_data) - len(train_data) // batch_size * batch_size  # len(data)=238400   batch_size=128   discard_n = 64
    train_data = np.delete(train_data, range(discard_n), axis=0)  # 238400删除去64，因为238336可以被128整除
    GT_data = np.delete(GT_data, range(discard_n), axis=0)
    print('^_^-training data and ground_truth data finished-^_^')
    return train_data, GT_data

def imsave(image, path):
    # return scipy.misc.imsave(path, image)
    return cv2.imwrite(path, np.uint8(image * 255))

if __name__ == '__main__':
    train_data, GT_data = datagenerator(train_data_dir='data/Train400-25', GT_data_dir='data/Train400')
    plt.figure()
    plt.imshow(train_data[525].reshape(40, 40))
    plt.figure()
    plt.imshow(GT_data[525].reshape(40, 40))
    print(train_data.shape)
    print(GT_data.shape)
