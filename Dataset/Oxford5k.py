import os
import numpy as np
from utils import Read_DogSIFT
from utils import Read_projection


class Oxford5k(object):
    def __init__(self, root_path=None, sift_path=None, projection_path=None, kp_path=None):
        assert sift_path is not None
        super(Oxford5k, self).__init__()
        self.root_path = root_path
        self.sift_path = sift_path
        self.kp_path = kp_path
        self.projection_path = projection_path
        self.name = 'Oxford5k'

        self.Image_names = os.listdir(root_path)
        self.Name_to_ID = {k: v for v, k in enumerate(self.Image_names)}

        self.N_images = len(self.Image_names)


    def DB_features(self):
        index = []
        Des_to_Im = []
        Descriptors = []
        projections = []
        kps = []
        idxs = []

        curr = 0
        for k, img_fn in enumerate(self.Image_names):
            des = np.load(os.path.join(self.sift_path, img_fn + '.sift.npy'))
            Descriptors.append(des)
            index.append(len(des))
            idxs.append(curr)
            curr += len(des)
            Des_to_Im.extend([k]*len(des))
        idxs.append(curr)
        Descriptors = np.concatenate(Descriptors, axis=0)
        print('curr==len(Descriptors)', curr == len(Descriptors))
        index = np.array(index)
        idxs = np.array(idxs)
        Des_to_Im = np.array(Des_to_Im)

        for img_fn in self.Image_names:
            projection = np.load(os.path.join(self.projection_path, img_fn + '.proj.npy'))
            projections.append(projection)
        projections = np.concatenate(projections, axis=0)

        for img_fn in self.Image_names:
            kp = np.load(os.path.join(self.kp_path, img_fn + '.kp.npy'))
            kps.append(kp)
        kps = np.concatenate(kps, axis=0)

        Descriptor_IDs = np.arange(Descriptors.shape[0])  # 就是从0开始的标号
        print('features total num:{},dim:{}'.format(Descriptors.shape[0], Descriptors.shape[1]))
        print('>>testing...')
        for i in range(5):
            des1 = Descriptors[idxs[i]: idxs[i + 1]]
            des2 = np.load(os.path.join(self.sift_path, self.Image_names[i] + ".sift.npy"))
            print(des1.shape)
            print(des2.shape)
            print((des1 == des2).sum() == des1.size)

        return index, Descriptors, projections, kps, Descriptor_IDs, Des_to_Im, idxs
