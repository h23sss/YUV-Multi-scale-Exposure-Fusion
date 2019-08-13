import weighting_map
import skimage.io as io
import skimage.color
import cv2
import numpy as np
import exp_fusion
import Gaussian_pyramid_fusion
import scipy.misc
from skimage.util import img_as_ubyte
from detail import Detail_extraction
if __name__ == "__main__":

    img = io.ImageCollection('pictures/*.jpg').concatenate()
    img = skimage.color.rgb2yuv(img)
    # img = img_as_ubyte(img)
    #RGB转YUV
    img_num = img.shape[0]
    imU = img[...,1]
    imV = img[...,2]
    imY = img[...,0]

    Ld=Detail_extraction(imY)

    we = weighting_map.weight_map(img,imY,imU,imV).transpose(2,0,1)  # 权重图的构造

    # 归一化
    we = we + 1e-12
    W_sum = np.tile(np.sum(we,0),(img_num,1,1))
    w = we / W_sum
    w = w.transpose(1,2,0)


    Y = exp_fusion.fusion(img, w, imY)  # Y曝光融合
    # cv2.imshow('1',imY[1])
    # cv2.imshow('2',Y)
    U = Gaussian_pyramid_fusion.fusion(imU, w)  # U通道基于高斯金字塔融合算法
    # io.imshow(imU[0])
    # io.show()
    # io.imshow(U)
    # io.show()
    V = Gaussian_pyramid_fusion.fusion(imV, w)  # V通道基于高斯金字塔融合算法
    # io.imshow(imV[0])
    # io.show()
    # io.imshow(V)
    # io.show()
    # merged = cv2.merge([Y, U, V])
    merged = np.zeros((Y.shape[0],Y.shape[1],3))
    merged[:, :, 0] = Y
    merged[:, :, 1] = U
    merged[:, :, 2] = V

    cv2.imshow('1',Ld)
    Y = np.multiply(Y,np.exp2(Ld))

    e=skimage.color.yuv2rgb(merged)

    # io.imshow(e)
    # io.show()
    # cv2.imshow('3', e[:, :, ::-1])
    scipy.misc.imsave(name='out.jpg',arr=e)
    cc=1
    # e = cv2.cvtColor(merged.astype(np.float32), cv2.COLOR_YUV2BGR)
    # cv2.imshow('7.jpg', e)
    # # cv2.imwrite('1.jpg', merged)
    # cv2.waitKey()
    # cv2.destroyAllWindows()