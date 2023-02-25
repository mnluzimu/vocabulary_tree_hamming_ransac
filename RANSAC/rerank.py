import pydegensac
import numpy as np
from copy import deepcopy


def verify_pydegensac(kps1, kps2, tentatives, th=4.0, n_iter=2000, is_H=False):
    src_pts = np.float32([kps1[m.queryIdx] for m in tentatives]).reshape(-1, 2)
    dst_pts = np.float32([kps2[m.trainIdx] for m in tentatives]).reshape(-1, 2)
    if is_H:
        H, mask = pydegensac.findHomography(src_pts, dst_pts, th, 0.99, n_iter)
    else:
        H, mask = pydegensac.findFundamentalMatrix(src_pts, dst_pts, th, 0.99, n_iter)

    num = int(deepcopy(mask).astype(np.float32).sum())
    print('pydegensac found {} inliers'.format(num))
    return H, mask, num

