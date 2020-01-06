import os

import cv2
import numpy as np


from HOG_SVM.detect_bounding_boxes import HOG_SVM_detector


'''
对于原始数据集，使用 detect_bounding_boxes.py 预测出每张图片的 bounding boxes 信息，
对于每张图片的每个目标，其预测边界框由 4 个 int 表达，分别是 (Xmin, Ymin) - (Xmax, Ymax) 
PS：一行 存储一张图片中的 所有 边界框坐标，Test：741 行
以 .npy 格式保存在 
./predict
    HOG_SVM.npy
'''


def solve(detector, path, save_file_name):
    original_image_path = './HOG_SVM/data/Test/' # 这个没有标签干扰
    save_path = "./predict/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    original_image_files = os.listdir(original_image_path)
    save_predict_bb = []
    cnt = 0
    for file in original_image_files:
        image_path = original_image_path + file
        predict_bb = detector(image_path, path)
        save_predict_bb.append(predict_bb)
        cnt += 1
        print(cnt, " ", file, " has done")

    save_predict_bb = np.array(save_predict_bb)
    np.save(save_path+save_file_name, save_predict_bb)
    # che = np.load(save_path+save_file_name)
    # for i in range(che.shape[0]):
    #     if che[i] == save_predict_bb[i]:
    #         continue
    #     else :
    #         print("error: ", i)

if __name__ == '__main__':
    solve(HOG_SVM_detector, './HOG_SVM/model/lin_svm_clf_hard_examples.pkl', 'HOG_SVM_hard_examples.npy')
