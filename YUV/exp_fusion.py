import numpy as np
from skimage.transform import pyramid_gaussian
import cv2

def Gaussian_pyramid(I, n):  # 高斯金字塔
    gau = []
    for (i, resized) in enumerate(pyramid_gaussian(I, downscale=2, sigma=3)):
        if i == n:
            break
        gau.append(resized)
    return gau


def Laplacian_pyramid(I, n):  # 拉普拉斯金字塔
    lap = []
    gau = Gaussian_pyramid(I, n)
    for i in range(0, n - 1):
        temp = gau[i] - cv2.resize(gau[i + 1], (gau[i].shape[1], gau[i].shape[0]))
        lap.append(temp)
    lap.append(gau[n - 1])
    return lap


def Reconstruct_Laplacian_pyramid(I):  # 重构由Rl构成的拉普拉斯金字塔，得到最终的融合图像R
    n = len(I)
    R = I[n-1]
    for i in range(n - 1, -1, -1):
        R = I[i] + cv2.resize(R, (I[i].shape[1], I[i].shape[0]))
    return R


def fusion(I, W, Y):
    img_shape = I[0].shape
    h = img_shape[0]
    w = img_shape[1]
    num = len(I)
    n = int(np.log2(min(h, w))) - 2
    R = Gaussian_pyramid(np.zeros((h, w)), n)
    for i in range(0, num):  # Y曝光融合
        RnG = Gaussian_pyramid(W[:, :, i], n)
        RnL = Laplacian_pyramid(Y[i], n)
        for j in range(0, n):
            if (j == n - 1):
                LL = Laplacian_pyramid(cv2.resize(RnL[j], (w, h)),1)
                L = cv2.resize(LL[0], (RnL[j].shape[1], RnL[j].shape[0]))
                R[j] = R[j] + np.multiply(RnG[j] + 1.5 * abs(L), RnL[j])
            else:
                R[j] = R[j] + np.multiply(RnG[j], RnL[j])
    RY = Reconstruct_Laplacian_pyramid(R)
    RY = RY*255.0

    return RY
