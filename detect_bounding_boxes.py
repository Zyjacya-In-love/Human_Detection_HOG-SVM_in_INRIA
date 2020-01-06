import time
import imutils
import os
import cv2
from sklearn.externals import joblib
from skimage.io import imread, imsave, imshow
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from skimage.color import rgb2gray
from imutils.object_detection import non_max_suppression
import numpy as np
import matplotlib.pyplot as plt

'''
提供边界框检测函数 HOG_SVM_detector
对于被检测的图像使用滑动窗口法（sliding window）提取待选窗口（Proposal）
考虑到有不同大小的物体，那需要不同的尺寸的框来进行处理，使用图像金字塔（Image pyramids）解决不同尺度下的目标检测问题
构建一个尺度 scale=1.2 的图像金字塔，以及一个分别在x方向和y方向步长为 (5,5) 像素大小的滑窗
使用 图像金字塔+滑窗+分类器 可以获得若干个可能是行人的矩形框（待选窗口），
然后对这些待选窗口再进行非极大值抑制（Non-Maximum Suppression），过滤掉一些重合程度较高的待选窗口，
最终就获得了作为候选区域的 ROI。
返回 predict_bounding_box，一行存储一个 bb，bb 由 4 个 int 表达，分别是 (Xmin, Ymin) - (Xmax, Ymax)
需要的参数有：
image_path 待检测图像的位置
model_path 训练好的模型位置 
'''

def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])

def HOG_SVM_detector(image_path, model_path):
    ori_img = cv2.imread(image_path)
    if type(model_path)==str:
        model = joblib.load(model_path)
    else:
        model = model_path
    # print(type(model_path)==str)
    # 将图像裁剪到最大宽度为400个像素，之所以降低我们的图像维度(其实就是之所以对我们的图像尺寸进行裁剪)主要有两个原因：
    # 1. 减小图像的尺寸可以减少在图像金字塔中滑窗的数目，如此可以降低检测的时间，从而提高整体检测的吞吐量。
    # 2. 调整图像的尺寸能够整体提高行人检测的精度。
    minAxis_Image = 400
    img = imutils.resize(ori_img, width=min(minAxis_Image, ori_img.shape[1]))  # 修改的图像是以最小的那个数字对齐
    # 用于之后复原图片
    HO, WO, _ = ori_img.shape
    HR, WR, _ = img.shape

    # 改 程序运行时间过长，将参数改大，注释中是 论文参数
    # 这个设置是按照开创性的 Dalal 和 Triggs 论文来设置的 Histograms of Oriented Gradients for Human Detection
    win_size = (64, 128) # 窗口的大小固定在64*128像素大小
    step_size = (5, 5) # 一个分别在x方向和y方向步长为(4,4)像素大小的滑窗
    downscale = 1.2 # 尺度 scale=1.05 的图像金字塔
    threshold = 0.6 # 设定 SVM 预测的阈值，即只有当预测的概率大于 0.6 时，才会确定支持向量机的预测
    overlapThresh = 0.3 # nms

    rectangles = []
    scores = []
    scale = 0
    for img_scale in pyramid_gaussian(img, downscale=downscale):
        if img_scale.shape[0] < win_size[1] or img_scale.shape[1] < win_size[0]:
            break

        for (x, y, img_window) in sliding_window(img_scale, win_size, step_size):
            if img_window.shape[0] != win_size[1] or img_window.shape[1] != win_size[0]:  # ensure the sliding window has met the minimum size requirement
                continue

            fd = hog(rgb2gray(img_window), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
            fd = fd.reshape(1, -1)
            predict = model.predict(fd)
            score = model.decision_function(fd)[0]
            if predict == 1 and score > threshold:
                x = int(x * (downscale ** scale))
                y = int(y * (downscale ** scale))
                w = int(win_size[0] * (downscale ** scale))
                h = int(win_size[1] * (downscale ** scale))
                rectangles.append((x, y, x+w, y+h))
                scores.append(score)
        scale += 1
    rectangles = np.array(rectangles)
    scores = np.array(scores)
    bounding_boxes = non_max_suppression(rectangles, probs=scores, overlapThresh=overlapThresh)

    # print("rectangles.shape : ", rectangles.shape)
    # print("bounding_boxes.shape : ", bounding_boxes.shape)
    # 如果检测到了内容
    if rectangles.shape[0] > 0:
        # 复原
        rectangles = rectangles.astype(float)
        rectangles[:, [0, 2]] *= (WO / WR)
        rectangles[:, [1, 3]] *= (HO / HR)
        rectangles = rectangles.astype(int)

        bounding_boxes = bounding_boxes.astype(float)
        bounding_boxes[:, [0, 2]] *= (WO / WR)
        bounding_boxes[:, [1, 3]] *= (HO / HR)
        bounding_boxes = bounding_boxes.astype(int)

    # print("rectangles.shape : ", rectangles.shape)
    # print("bounding_boxes.shape : ", bounding_boxes.shape)
    # print(bounding_boxes)
    # return rectangles, bounding_boxes
    return bounding_boxes


if __name__ == '__main__':
    Test_path = './data/Test/'
    file = 'person_075.png'
    image_path = Test_path+file
    model_path = './model/lin_svm_clf_hard_examples.pkl'

    # 计时
    start = time.clock()
    model = joblib.load(model_path)
    rectangles, bounding_boxes = HOG_SVM_detector(image_path, model)
    elapsed = (time.clock() - start)
    print("Time used:", elapsed)

    ori_img = cv2.imread(image_path)
    img_copy = ori_img.copy()


    for (x1, y1, x2, y2) in rectangles:
        cv2.rectangle(ori_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    for (x1, y1, x2, y2) in bounding_boxes:
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)


    # plt.axis("off")
    # plt.imshow(cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB))
    # plt.title("Raw Detection before NMS")
    # plt.show()
    #
    # plt.axis("off")
    # plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    # plt.title("Final Detections after applying NMS")
    # plt.show()
    cv2.imwrite("Before NMS.png", ori_img)
    cv2.imwrite("After NMS.png", img_copy)

    # cv2.imshow("Before NMS", ori_img)
    # cv2.imshow("After NMS", img_copy)
    # cv2.waitKey(0)
    # imshow(img)
    # imshow(img_copy)
    # plt.show()
    #
    # plt.axis("off")
    # plt.imshow(img)
    # plt.title("Raw Detection before NMS")
    # plt.show()

    # plt.axis("off")
    # plt.imshow(img_copy)
    # plt.title("Final Detections after applying NMS")
    # plt.show()
