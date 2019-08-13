import numpy as np
from scipy import signal

def la_filter(Y):
    laplacian_operator = list([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    C = signal.convolve2d(Y, laplacian_operator, mode="same")
    C = abs(C)
    return C


def contrast(Y, exposure_num, img_rows, img_cols):  # 对比度
    n = exposure_num
    C = np.zeros((img_rows, img_cols, n))
    for i in range(0, n):
        C[:, :, i] = la_filter(Y[i])
    return C


def saturation(U, V, exposure_num, img_rows, img_cols):
    n = exposure_num
    S = np.zeros((img_rows, img_cols, n))
    for i in range(0, n):
        S[:, :, i] = abs(U[i]) + abs(V[i]) + 1
    return S


def exposure(Y, exposure_num, img_rows, img_cols):
    n = exposure_num
    E = np.zeros((img_rows, img_cols, n))
    for i in range(0, n):
        E[:, :, i] = np.exp(-((Y[i]- 0.5) ** 2 / (2 * (0.2 ** 2))))
    return E


def quality_measure(Y,exposure_num, img_rows, img_cols):
    n = exposure_num
    B = np.zeros((img_rows, img_cols, n))

    for i in range(0, n):
        y = np.average(Y[i])
        B[:, :, i] = y ** 2
    return B


def weight_map(I,Y,U,V):
    img_shape = I[0].shape
    img_rows = img_shape[0]
    img_cols = img_shape[1]
    exposure_num = len(I)
    r = img_rows
    c = img_cols
    w = np.ones((r, c, exposure_num))
    w = np.multiply(w, contrast(Y, exposure_num, img_rows, img_cols))
    w = np.multiply(w, saturation(U,V, exposure_num, img_rows, img_cols))
    w = np.multiply(w, exposure(Y, exposure_num, img_rows, img_cols))
    w = np.multiply(w, quality_measure(Y, exposure_num, img_rows, img_cols))
    return w
