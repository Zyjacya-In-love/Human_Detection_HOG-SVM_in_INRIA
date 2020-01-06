import random
import os
import cv2
from skimage.io import imread, imsave, imshow
from skimage.feature import hog
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt

'''
1. 将 train_64*128_H96 和 test_64*128_H96 中 pos 和 neg 中的图像均设置成相同的大小，即 64*128 像素。
    对于正样本 pos 直接使用中心为 64x128 像素的窗口
    对负样本 neg 每个原始负样本随机生成 10 个窗口
2. 截取完成后直接都提取 hog 特征，并打好标签

以 .npy 格式保存 在 
./feature/
    test.npy 5662*1780 （1126 + 453*10）
    train.npy 14596*3780 (2416 + 1218*10)
'''

def solve(mode):
    pos_img_path = './INRIAPerson/{}_64x128_H96/pos/'.format(mode)
    neg_img_path = './INRIAPerson/{}_64x128_H96/neg/'.format(mode)
    save_path = "./feature/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    hog_feature = []
    # /pos
    for file in os.listdir(pos_img_path):
        img = imread(pos_img_path+file)
        center = img.shape[0] // 2, img.shape[1] // 2
        crop_img = img[center[0]-64:center[0]+64, center[1]-32:center[1]+32, :]
        fd = hog(rgb2gray(crop_img), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        hog_feature.append(fd)
    pos_num = len(hog_feature)
    # /neg
    for file in os.listdir(neg_img_path):
        img = imread(neg_img_path + file)
        # 图片大小应该能能至少包含一个 64 * 128 的窗口
        if img.shape[0] >= 128 and img.shape[1] >= 64:
            for i in range(10):
                x = random.randint(0, img.shape[0] - 128)  # 左上角x坐标
                y = random.randint(0, img.shape[1] - 64)  # 左上角y坐标
                crop_img = img[x:x + 128, y:y + 64, :]
                fd = hog(rgb2gray(crop_img), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
                hog_feature.append(fd)
    hog_feature = np.array(hog_feature)
    label = np.zeros(hog_feature.shape[0])
    label[0:pos_num] = 1
    data = {'X':hog_feature, 'Y':label}
    np.save(save_path + mode + '.npy', data)
    che = np.load(save_path + mode + '.npy', allow_pickle=True)
    che = che.item()
    X = che['X']
    Y = che['Y']
    # print(X.shape)
    # print(Y.shape)
    if (X == hog_feature).all() and (Y == label).all():
        print("check YES")


if __name__ == '__main__':
    solve('train')
    solve('test')
