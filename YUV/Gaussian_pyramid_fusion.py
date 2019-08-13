import cv2
import numpy as np
import math


def gaussian_kernel(size=5, sigma=0.4):
    return cv2.getGaussianKernel(ksize=size, sigma=sigma)


def image_reduce(image):
    kernel = gaussian_kernel()
    out_image = cv2.filter2D(image, -1, kernel)
    height, width = image.shape[:2]
    width = int(width * 0.5)
    height = int(height * 0.5)
    size = (width, height)
    out_image = cv2.resize(out_image, size)
    return out_image


def image_expand(image, model):
    kernel = gaussian_kernel()
    height, width = model.shape[:2]
    size = (width, height)
    out_image = cv2.resize(image, size)
    out_image = cv2.filter2D(out_image, -1, kernel)
    return out_image


def gaussian_pyramid(img, depth):
    G = img.copy()
    gp = [G]
    for i in range(depth):
        G = image_reduce(G)
        gp.append(G)
    return gp


def laplacian_pyramid(img, depth):
    gp = gaussian_pyramid(img, depth)
    lp = [gp[depth - 1]]
    for i in range(depth - 1, 0, -1):
        GE = image_expand(gp[i], gp[i - 1])
        L = cv2.subtract(gp[i - 1], GE)
        lp = [L] + lp
    return lp


def pyramid_collapse(pyramid):
    depth = len(pyramid)
    collapsed = pyramid[depth - 1]
    for i in range(depth - 2, -1, -1):
        collapsed = cv2.add(image_expand(collapsed, pyramid[i]), pyramid[i])
    return collapsed


def fusion(images, weights):
    img_shape = images[0].shape
    h = img_shape[0]
    w = img_shape[1]
    depth = int(np.log2(min(h, w))) - 2
    num = len(images)
    # compute pyramids
    lps = []
    gps = []
    for i in range(0, num):
        lps.append(laplacian_pyramid(images[i], depth))
        gps.append(gaussian_pyramid(weights[:, :, i], depth))

    # combine pyramids with weights
    LS = []
    for l in range(depth):
        ls = np.zeros(lps[0][l].shape, dtype=np.uint8)
        for k in range(len(images)):
            lp = lps[k][l]
            gp = np.float32(gps[k][l]) / 255
            lp_gp = np.multiply(lp, gp)
            ls = ls + lp_gp
        LS.append(ls)

    # collapse pyramid
    fusion = pyramid_collapse(LS)
    fusion = fusion * 255
    return fusion
