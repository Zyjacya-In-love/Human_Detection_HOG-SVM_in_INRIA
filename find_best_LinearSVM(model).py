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
使用 SVM 分类，用 k 折交叉验证随机分出 训练集 和 验证集 以找出效果最佳的模型（超参数）
k 折交叉验证通过对 k 个不同分组训练的结果进行平均来减少方差，因此模型的性能对数据的划分就不那么敏感。
'''

def get_data_tar(mode):
    feature_path = './feature/{}.npy'.format(mode)
    data = np.load(feature_path, allow_pickle=True)
    data = data.item()
    return data['X'], data['Y']


# 不加难例
def find_best_svm(C_pool, train_X, train_y):
    diff_c_accuracy_scores = []
    # 藉由迭代的方式来计算不同参数对模型的影响，并返回交叉验证后的平均准确率
    for c in C_pool:
        lin_clf = svm.LinearSVC(C=c)
        scores = cross_val_score(lin_clf, train_X, train_y, cv=10, scoring='accuracy')
        diff_c_accuracy_scores.append(scores.mean())
        print("C = ", c, " accuracy = ", diff_c_accuracy_scores[-1])
    best_C = C_pool[np.argmax(diff_c_accuracy_scores)]
    return best_C, diff_c_accuracy_scores


'''
# 每次加难例 时间太长 舍弃
def find_best_svm(C_pool, train_X, train_y):
    diff_c_accuracy_scores = []
    # 藉由迭代的方式来计算不同参数对模型的影响，并返回交叉验证后的平均准确率
    for c in C_pool:
        lin_clf = svm.LinearSVC(C=c)
        # scores = cross_val_score(lin_clf, train_X, train_y, cv=10, scoring='accuracy')
        scores = []
        kf = KFold(n_splits=10)
        for train_index, valid_index in kf.split(train_X):
            now_train_X = train_X[train_index]
            now_train_y = train_y[train_index]
            clf = lin_clf.fit(now_train_X, now_train_y)
            hes = hard_examples(clf)
            X = np.concatenate((now_train_X, hes), axis=0)
            y = np.concatenate((now_train_y, np.zeros(hes.shape[0])), axis=0)
            clf = lin_clf.fit(X, y)
            acc = clf.score(train_X[valid_index], train_y[valid_index])
            scores.append(acc)
        scores = np.array(scores)
        diff_c_accuracy_scores.append(scores.mean())
        print("C = ", c, " accuracy = ", diff_c_accuracy_scores[-1])
    best_C = C_pool[np.argmax(diff_c_accuracy_scores)]
    return best_C, diff_c_accuracy_scores
'''

if __name__ == '__main__':
    train_X, train_y = get_data_tar('train')
    # 因为每张图片提取出来的 HOG 特征有 3780 维，所以我们使用线性 SVM 就足够可分
    C_pool = np.arange(0.001, 0.021, 0.001)

    best_C, diff_c_accuracy_scores = find_best_svm(C_pool, train_X, train_y)
    np.save('./accuracy_.npy', diff_c_accuracy_scores)

    # 以下代码用于 绘图寻找最优模型参数 C，以及训练保存最优 SVM 模型

    # diff_c_accuracy_scores = np.load('./accuracy.npy')

    # 可视化数据
    # figure, ax = plt.subplots(figsize=(6, 5))
    #
    # font = {'family': 'Times New Roman',
    #          'weight': 'normal',
    #          'size': 15,}
    # plt.grid(linestyle="--")
    # # 设置坐标刻度值的大小以及刻度值的字体
    # plt.tick_params(labelsize=15)
    # labels = ax.get_xticklabels() + ax.get_yticklabels()
    # [label.set_fontname('Times New Roman') for label in labels]
    # plt.plot(C_pool, diff_c_accuracy_scores)
    # plt.xticks(C_pool, rotation=45)
    # plt.xlabel('Value of C for LinearSVC', font)
    # plt.ylabel('Cross-Validated Accuracy', font)
    # plt.show()

    # 全部数据训练最优 SVM 模型，用测试数据测试 最佳模型
    best_C = C_pool[np.argmax(diff_c_accuracy_scores)]
    print("best_C : ", best_C)
    best_lin_svm_clf = svm.LinearSVC(C=best_C)
    best_lin_svm_clf.fit(train_X, train_y)  # 训练模型
    # hes = hard_examples(best_lin_svm_clf)
    # X = np.concatenate((train_X, hes), axis=0)
    # y = np.concatenate((train_y, np.zeros(hes.shape[0])), axis=0)
    # best_lin_svm_clf.fit(X, y)
    test_X, test_y = get_data_tar('test')
    print("test : ", best_lin_svm_clf.score(test_X, test_y))  # 看看评分

    # 保存最佳模型
    save_path = './model/'
    save_file_name = 'lin_svm_clf.pkl'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    joblib.dump(best_lin_svm_clf, save_path+save_file_name, compress=3)  # 保存训练好的clf模型 compress读取速度
    che_clf = joblib.load(save_path+save_file_name)  # 读取训练好的clf模型
    print("check : ", che_clf.score(test_X, test_y))  # 测试评分
