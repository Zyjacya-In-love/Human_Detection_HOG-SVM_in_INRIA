import random

import cv2
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.feature import hog
import numpy as np
import os
from sklearn import svm
from sklearn.model_selection import cross_val_score # K折交叉验证模块
import matplotlib.pyplot as plt #可视化模块
from sklearn.externals import joblib
from sklearn.model_selection import KFold

from detect_bounding_boxes import HOG_SVM_detector

'''
hard-negative mining
利用model进行负样本难例检测。
对 Train 里的负样本进行多尺度检测，如果分类器误检出非目标则截取图像加入负样本中。
结合难例重新训练model。
'''

def get_data_tar(mode):
    feature_path = './feature/{}.npy'.format(mode)
    data = np.load(feature_path, allow_pickle=True)
    data = data.item()
    return data['X'], data['Y']

def hard_examples(clf):
    Train_neg_path = './data/Train/'
    hog_feature = []
    cnt = 0
    for file in os.listdir(Train_neg_path):
        cnt += 1
        image_path = Train_neg_path + file
        # image_path = './data/Train/00001042a.png'
        print(cnt, " ", image_path)
        hard_bb = HOG_SVM_detector(image_path, clf)
        # print(np.shape(hard_bb))
        if np.shape(hard_bb)[0] == 0:
            continue
        img = imread(image_path)
        # print("img.shape: ", img.shape)
        for one in hard_bb:
            # print(one)
            crop_img = img[one[1]:one[3], one[0]:one[2] :]
            # print(crop_img)
            fd = hog(cv2.resize(rgb2gray(crop_img), (64,128)), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
            hog_feature.append(fd)
    return np.array(hog_feature)


if __name__ == '__main__':
    train_X, train_y = get_data_tar('train')
    test_X, test_y = get_data_tar('test')
    model_path = './model/lin_svm_clf.pkl'
    best_lin_svm_clf = joblib.load(model_path)  # 读取训练好的clf模型
    hes = hard_examples(best_lin_svm_clf)
    X = np.concatenate((train_X, hes), axis=0)
    y = np.concatenate((train_y, np.zeros(hes.shape[0])), axis=0)
    best_lin_svm_clf.fit(X, y)
    print("test : ", best_lin_svm_clf.score(test_X, test_y))  # 看看评分
    # 保存最佳模型
    save_path = './model/'
    save_file_name = 'lin_svm_clf_hard_examples.pkl'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    joblib.dump(best_lin_svm_clf, save_path + save_file_name, compress=3)  # 保存训练好的clf模型 compress读取速度
