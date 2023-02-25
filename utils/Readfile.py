import numpy as np


def Read_DogSIFT(path):  # path对应 .jpg.sift文件
    SIFT_features = np.fromfile(path, np.int8)
    SIFT_features = SIFT_features.reshape((-1, 128))
    SIFT_Nums = len(SIFT_features)
    print('Features Path:{},Num:{},Dim:{}'.format(path, SIFT_Nums, 128))
    return SIFT_features


def Read_projection(path):
    projections = np.fromfile(path, np.int8)
    projections = projections.reshape((-1, 64))
    projection_Nums = len(projections)
    print('projection Path:{},NUM:{},Dim:{}'.format(path, projection_Nums, 64))
    return projections


