import numpy as np
import scipy.linalg as linalg


def cal_Gradh(I):
    Gradh = np.array([[I[i, j + 1] - I[i, j] for j in range(I.shape[1] - 1)] for i in range(I.shape[0])])
    Gradh = np.insert(Gradh, I.shape[1] - 1, values=0, axis=1)
    return Gradh

def cal_Gradv(I):
    Gradv = np.array([[I[i + 1, j] - I[i, j] for j in range(I.shape[1])] for i in range(I.shape[0] - 1)])
    Gradv = np.insert(Gradv, I.shape[0] - 1, values=0, axis=0)
    return Gradv

def  Vector_field(imY,xita1=127,xitaN=127):
    Y1 = imY[np.argmin(np.average(imY, axis=(1, 2)))]
    YN = imY[np.argmax(np.average(imY, axis=(1, 2)))]
    I1 = np.log2(Y1 + 1)
    IN = np.log2(YN + 1)
    Gradh1 = cal_Gradh(I1)
    GradhN = cal_Gradh(IN)
    Gradv1 = cal_Gradv(I1)
    GradvN = cal_Gradv(IN)
    T1 = np.where(Y1 < xita1, Y1 + 1, np.where(xita1 + 1 - 16 * (Y1 - xita1) < 0, 0, xita1 + 1 - 16 * (Y1 - xita1)))
    TN = np.where(YN > xitaN, 256 - YN, np.where(256 - xitaN + 16 * (YN - xitaN) < 0, 0, 256 - xitaN + 16 * (YN - xitaN)))
    vh = np.zeros(Gradh1.shape)
    vv = np.zeros(Gradv1.shape)
    for i in range(vh.shape[0]):
        for j in range(vh.shape[1]-1):
            Th = T1[i, j] * T1[i, j + 1] + TN[i, j] * TN[i, j + 1]
            if Th > 0: vh[i, j] = (T1[i, j] * T1[i, j + 1] * Gradh1[i, j] + TN[i, j] * TN[i, j + 1] * GradhN[i, j]) / Th
    for i in range(vv.shape[0]-1):
        for j in range(vv.shape[1]):
            Tv = T1[i, j] * T1[i + 1, j] + TN[i, j] * TN[i + 1, j]
            if Tv > 0: vv[i, j] = (T1[i, j] * T1[i + 1, j] * Gradv1[i, j] + TN[i, j] * TN[i + 1, j] * GradvN[i, j]) / Tv
    return vh,vv

def yita(v,gama=2.0,yipuxinong=2.0):
    return 1.0/((abs(v)**gama+yipuxinong)**0.5)

def  Detail_extraction(imY):

    vh,vv=Vector_field(imY)
    tmp = np.array([[yita(x) for x in v] for v in vh])
    Avh = np.diag(np.diag(tmp))
    DAvh = cal_Gradh(Avh)
    DAvhD = cal_Gradh(DAvh)
    tmp = np.array([[yita(x) for x in v] for v in vv])
    Avv = np.diag(np.diag(tmp))
    DAvv = cal_Gradh(Avv)
    DAvvD = cal_Gradh(DAvv)
    iden =np.identity(Avv.shape[0])
    Ld = np.zeros(imY[0].shape)
    Ld = linalg.solve(iden+DAvhD+DAvvD, np.dot(DAvhD,vh)+np.dot(DAvvD,vv))
    return Ld



