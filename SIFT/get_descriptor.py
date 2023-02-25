import cv2 as cv
import os
import numpy as np
from utils import Read_DogSIFT
import os


def get_sift(name, imgs_path, outPath_des, outPath_proj, outPath_kp, P_path, projection=True):
    pic = cv.imread(os.path.join(imgs_path, name))
    sift = cv.xfeatures2d.SIFT_create(nfeatures=500, contrastThreshold=0.01, edgeThreshold=30, sigma=1.6)
    kps, des = sift.detectAndCompute(pic, None)
    des_np = des.astype(np.uint8)
    kps = np.array([x.pt for x in kps]).astype(np.float16)

    outPath_kp = os.path.join(outPath_kp, name + '.kp')
    outPath_des = os.path.join(outPath_des, name + '.sift')

    np.save(outPath_kp, kps)
    np.save(outPath_des, des_np)

    if projection is True:
        P = np.load(P_path)
        outPath_proj = os.path.join(outPath_proj, name + '.proj')
        proj_np = np.transpose(np.matmul(P, np.transpose(des_np))).astype(np.int8)
        np.save(outPath_proj, proj_np)
    


if __name__ == "__main__":
    imgs_path = r"F:\\science_research\\image_retrieval\\dataset\\oxford5k\\oxbuild_images-v1"
    s = os.listdir(imgs_path)

    outPath_des = "..\oxford5k\oxford5k_sift"
    outPath_proj = "..\oxford5k\oxford5k_projection"
    outPath_kp = "..\oxford5k\oxford5k_kp"
    P_path = "..\oxford5k\P.npy"

    if not os.path.exists(outPath_des):
        os.makedirs(outPath_des)
    if not os.path.exists(outPath_proj):
        os.makedirs(outPath_proj)
    if not os.path.exists(outPath_kp):
        os.makedirs(outPath_kp)

    
    d = 128
    d_b = 64
    M = np.random.randn(d, d)
    Q, R = np.linalg.qr(M)
    P = Q[:d_b, :]
    np.save(P_path, P)
    
    cnt = 0
    for name in s:
        get_sift(name, imgs_path, outPath_des, outPath_proj, outPath_kp, P_path, projection=True)
        cnt = cnt + 1
        if cnt % 10 == 0:
            print('\r>>{}/{}'.format(cnt, len(s)), end='')


