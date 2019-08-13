import weighting_map
import cv2
import os
import numpy as np
import exp_fusion
import Gaussian_pyramid_fusion

if __name__ == "__main__":

    img = []
    for filename in os.listdir(r"./pictures"):  # 读取文件图片
        filenames = './pictures/' + filename
        print(filenames)
        img.append(cv2.imread(filenames, 1))

    #RGB转YUV
    img_num = len(img)
    imU = []
    imV = []
    imY = []
    for i in range(0,img_num):
        temp=cv2.cvtColor(img[i],cv2.COLOR_BGR2YUV)
        y,u,v=cv2.split(temp)
        imY.append(y), imU.append(u), imV.append(v)

    w = weighting_map.weight_map(img,imY,imU,imV)  # 权重图的构造

    # 归一化
    w = w + 1e-12
    W_sum = []
    for i in range(img_num):
        W_sum.append(np.sum(w, 2))
    W_sum = np.array(W_sum)
    W_sum = W_sum.swapaxes(0, 2)
    W_sum = W_sum.swapaxes(0, 1)
    w = w / W_sum

    Y = exp_fusion.fusion(img, w, imY)  # Y曝光融合

    U = Gaussian_pyramid_fusion.fusion(imU, w)  # U通道基于高斯金字塔融合算法

    V = Gaussian_pyramid_fusion.fusion(imV, w)  # V通道基于高斯金字塔融合算法

    merged = cv2.merge([Y, U, V])
    e = cv2.cvtColor(merged, cv2.COLOR_YUV2BGR)
    cv2.imshow('7.jpg', e)
    # cv2.imwrite('1.jpg', merged)
    cv2.waitKey()
    cv2.destroyAllWindows()