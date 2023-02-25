from glob import glob
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import math
import os
import pickle
import time
import utils
import cv2
from RANSAC import verify_pydegensac


class tree_node(object):
    def __init__(self):
        super(tree_node, self).__init__()
        self.weight = None
        self.children = None
        self.Descriptor_IDs = None
        self.InvertedIndex = None  # store how many descriptors each image have in this node
        self.Centroid = None  # center of cluster
        self.index = None  # the index of this node
        self.median_vec = None  # the median_vec to be used in making bit vector


class VocTree(object):
    def __init__(self, Dataset=None,
                 Tree_path=None,
                 BoFs_path=None,
                 bitVec_path=None,
                 des2Im_path=None,
                 Feature_type='HAsift',
                 Train=True,
                 branchs=10,
                 maximum_height=6):
        super(VocTree, self).__init__()

        self.Dataset = Dataset
        self.BoFs_path = BoFs_path
        self.Tree_path = Tree_path
        self.bitVec_path = bitVec_path
        self.des2Im_path = des2Im_path
        self.Train = Train  # 决定是否训练建树
        self.branchs = branchs
        self.maximum_height = maximum_height
        self.Feature_type = Feature_type  # 确定是 Dsift 还是 HAsift
        self.index, self.Descriptors, self.projections, self.kps, self.Descriptor_IDs, self.Des_to_Im, self.idxs = self.Dataset.DB_features()

        assert Feature_type in ['Dsift', 'HAsift']

        if Train:
            self.TrainingPhase(branchs, maximum_height)
        else:
            self.online()

    def TrainingPhase(self, branchs, maximum_height):
        since = time.time()

        # Create training data , index is where each picture begin or end
        self.bit_vec = [0 for i in range(len(self.Descriptors))]
        self.node_index = 0  # record the index of current node

        # Build VocTree
        Tree = self.BuildVocabularyTree(self.Descriptor_IDs, 0, branchs, maximum_height)

        # Build InvertedIndex
        self.BuildInvertedIndex(Tree)

        # Compute idf-weiget And Compute Dataset BoFs
        self.ComputeBoFs(Tree)  # BoF is a vector denoting each feature, each picture has such a vector to mark them

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        self.Tree = Tree

        # Store the Tree and the dataset BoFs for query
        self.save()

    def online(self):
        if os.path.exists(self.Tree_path) and os.path.exists(self.BoFs_path) and os.path.exists(self.bitVec_path) and os.path.exists(self.des2Im_path):
            with open(self.Tree_path, 'rb') as file:
                self.Tree = pickle.load(file)
            with open(self.BoFs_path, 'rb') as file:
                self.BoFs = pickle.load(file)
            with open(self.bitVec_path, 'rb') as file:
                self.bit_vec = pickle.load(file)
            with open(self.des2Im_path, 'rb') as file:
                self.Des_to_Im = pickle.load(file)
            print('Dataset is prepared')
            print('Dataset total num:{}'.format(self.Dataset.N_images))
        else:
            self.TrainingPhase(self.branchs, self.maximum_height)

    def ComputeClusters(self, Descriptor_IDs, branchs):
        Model = MiniBatchKMeans(branchs,
                                init='k-means++',
                                n_init=10,
                                tol=0.0001,
                                verbose=1)
        Model.fit(self.Descriptors[Descriptor_IDs])
        ChildDescriptor_IDs = [[] for i in range(branchs)]  # a 2 dim list built to save the ids in the child branches
        for i, ID in enumerate(Descriptor_IDs):
            ChildDescriptor_IDs[Model.labels_[i]].append(ID)  # labels_ stores the labels of the entries, by doing
            # this the ids are stored in corresponding branch
        return ChildDescriptor_IDs, Model.cluster_centers_

    def BuildVocabularyTree(self, Descriptor_IDs, current_level, branchs, maximum_height):
        if len(Descriptor_IDs) < branchs or current_level >= maximum_height:
            leaf_node = tree_node()  # maximum_height reached, make leaf node
            leaf_node.index = self.node_index
            self.node_index += 1
            leaf_node.Descriptor_IDs = Descriptor_IDs  # store descriptor ids

            leaf_node.median_vec = np.median(self.projections[Descriptor_IDs], axis=0)
            for i in Descriptor_IDs:
                bit_vec = np.where(self.projections[i] > leaf_node.median_vec, 1, 0).astype(np.int8)
                self.bit_vec[i] = utils.arr2bit(bit_vec, 64)

            return leaf_node
        else:
            Clusters, Centers = self.ComputeClusters(Descriptor_IDs, branchs)
            branch_node = tree_node()
            branch_node.index = self.node_index  # mark index of node
            branch_node.Descriptor_IDs = Descriptor_IDs  # store descriptor ids
            self.node_index += 1
            branch_node.children = []
            for i, C in enumerate(Clusters):  # build tree recursively
                Child = self.BuildVocabularyTree(C, current_level + 1, branchs, maximum_height)
                Child.Centroid = Centers[i]
                branch_node.children.append(Child)
            return branch_node

    def FindLeaf(self, root_node, Descriptor):  # recursively search for the leaf the descriptor belongs to
        if root_node.children is None:
            return root_node
        min_ = np.float('+inf')
        for child in root_node.children:
            dist = np.linalg.norm(Descriptor - child.Centroid)
            if min_ > dist:
                min_ = dist
                goto = child
        return self.FindLeaf(goto, Descriptor)

    def BuildInvertedIndex(self, root_node):
        index = 0
        for ID, length in enumerate(self.index):
            print('Image ID:{},Name:{}'.format(ID, self.Dataset.Image_names[ID]))

            image_Descriptors = self.Descriptors[index:index + length]  # index descriptors for this image
            index += length
            for Descriptor in image_Descriptors:
                leaf_node = self.FindLeaf(root_node, Descriptor)

                if leaf_node.InvertedIndex is None:
                    # invertedIndex stores the number of descriptors that fit this leaf node in each image
                    leaf_node.InvertedIndex = {}
                    leaf_node.InvertedIndex[ID] = 1
                else:
                    leaf_node.InvertedIndex[ID] = leaf_node.InvertedIndex.get(ID, 0) + 1

    def Merge_InvertedIndexes(self, InvertedIndexes_1, InvertedIndexes_2):
        # merge the two invertedIndex dicts together: same number index added
        if InvertedIndexes_1 is None:
            return InvertedIndexes_2
        if InvertedIndexes_2 is None:
            return InvertedIndexes_1

        InvertedIndexes = {}

        for key, value in InvertedIndexes_2.items():
            InvertedIndexes[key] = InvertedIndexes.get(key, 0) + value
        for key, value in InvertedIndexes_1.items():
            InvertedIndexes[key] = InvertedIndexes.get(key, 0) + value
        return InvertedIndexes

    def BuildVirtual_InvertedIndexes(self, root_node):
        # bof is likely not needed in hamming embedding, need to find out what voctree needs, about the
        # need of none-leaf nodes in voting.
        # then there is the problem of making a matching between descriptor id and image id, which can be used in voting
        # just index the image through the descriptors and vote accordingly
        print('-' * 50)
        print('Node index:{}'.format(root_node.index))
        if root_node.children is None:
            if root_node.InvertedIndex is None:
                root_node.weight = 0
            else:
                # get weight of leaf node
                root_node.weight = math.log10(self.Dataset.N_images / len(root_node.InvertedIndex))
                # len(root_node.InvertedIndex) shows how many images have descriptor in this node
                for image_ID, num in root_node.InvertedIndex.items():
                    # self.BoFs is a dict of dicts, stores in each image, what are the weight of each node
                    if image_ID in self.BoFs:
                        self.BoFs[image_ID][root_node.index] = root_node.weight * num
                    else:
                        self.BoFs[image_ID] = {}
                        self.BoFs[image_ID][root_node.index] = root_node.weight * num
            return root_node.InvertedIndex
        else:
            # it would seem here that both leafs and inside nodes have been considered in voting
            Virtual_InvertedIndexes = None
            # merge all child-node invertedIndexes, recursively, this is where recursion is made as well
            for child_node in root_node.children:
                Virtual_InvertedIndexes = self.Merge_InvertedIndexes(Virtual_InvertedIndexes,
                                                                     self.BuildVirtual_InvertedIndexes(child_node))

            if Virtual_InvertedIndexes is None:
                root_node.weight = 0
            else:
                root_node.weight = math.log10(self.Dataset.N_images / len(Virtual_InvertedIndexes))

                for image_ID, num in Virtual_InvertedIndexes.items():
                    if image_ID in self.BoFs:
                        self.BoFs[image_ID][root_node.index] = root_node.weight * num
                    else:
                        self.BoFs[image_ID] = {}
                        self.BoFs[image_ID][root_node.index] = root_node.weight * num
            return Virtual_InvertedIndexes

    def ComputeBoFs(self, root_node):
        self.BoFs = {}
        self.BoFs_norm = {}
        # cumpute BoF for each image
        _ = self.BuildVirtual_InvertedIndexes(root_node)

        for Image_ID, d in self.BoFs.items():
            # normalize BoFs, BoF: each image has a vector to show how weight on each feature
            norm = sum(abs(np.array(list(d.values()))))
            for node_ID in d:
                d[node_ID] /= norm
            self.BoFs[Image_ID] = d
            self.BoFs_norm[Image_ID] = norm

    def ComputeQBoF(self, Descriptors, root_node):
        # compute BoF of query image
        QBoF = {}
        for Descriptor in Descriptors:
            goto = root_node
            while goto.children is not None:
                min_ = np.float('+inf')
                for child in goto.children:
                    dist = np.linalg.norm(Descriptor - child.Centroid)
                    if min_ > dist:
                        min_ = dist
                        goto = child
                if goto.weight == 0:
                    continue
                QBoF[goto.index] = QBoF.get(goto.index, 0) + goto.weight

        norm = sum(abs(np.array(list(QBoF.values()))))

        for node_ID in QBoF:
            QBoF[node_ID] /= norm
        return QBoF

    def Query(self, Q_image_ID, root_node=None, BoFs=None, result_size=10):
        Image_Descriptors = self.Descriptors[self.idxs[Q_image_ID]:self.idxs[Q_image_ID + 1]]
        Q = self.ComputeQBoF(Image_Descriptors, root_node)
        Scores = np.zeros(self.Dataset.N_images)
        for Image_ID, d_vector in BoFs.items():
            score = 2
            for node_index, q in Q.items():
                if node_index in d_vector:
                    # compute score (? how to justify the method ?)
                    score = score + abs(q - d_vector[node_index]) - abs(q) - abs(d_vector[node_index])

            Scores[Image_ID] = score

        Rank_list = np.argsort(Scores)  # sort the pictures according to score
        return Rank_list[:result_size]

    def Query_with_Hamming(self, Q_image_ID,  root_node=None, result_size=10, ht=10):
        Scores = np.zeros(self.Dataset.N_images)
        Image_Descriptors = self.Descriptors[self.idxs[Q_image_ID]:self.idxs[Q_image_ID + 1]]
        Image_Projections = self.projections[self.idxs[Q_image_ID]:self.idxs[Q_image_ID + 1]]
        for Descriptor, Projection in zip(Image_Descriptors, Image_Projections):
            goto = root_node
            while goto.children is not None:
                min_ = np.float('+inf')
                for child in goto.children:
                    dist = np.linalg.norm(Descriptor - child.Centroid)
                    if min_ > dist:
                        min_ = dist
                        goto = child
                # if goto.weight == 0:
                #     continue
            bit_vec = np.where(Projection > goto.median_vec, 1, 0).astype(np.int8)
            bit_vec = utils.arr2bit(bit_vec, 64)
            for des_ID in goto.Descriptor_IDs:
                bit_vec_ds = self.bit_vec[des_ID]
                hd = np.sum(utils.bit2arr(bit_vec ^ bit_vec_ds, 64))
                # print(hd)
                if hd < ht:
                    # print(self.Des_to_Im[des_ID], Scores.get(self.Des_to_Im[des_ID], 0))
                    Scores[self.Des_to_Im[des_ID]] = Scores[self.Des_to_Im[des_ID]] + goto.weight ** 2
        Scores = - Scores
        Rank_list = Scores.argsort()  # sort the pictures according to score
        return Rank_list[:result_size]

    def reRank(self, rank_list, Q_image_ID, is_H=False):
        newScores = {}
        Q_Descriptors = self.Descriptors[self.idxs[Q_image_ID]:self.idxs[Q_image_ID + 1]]
        Q_kps = self.kps[self.idxs[Q_image_ID]:self.idxs[Q_image_ID + 1]]
        for img_ID in rank_list:
            kps = self.kps[self.idxs[img_ID]:self.idxs[img_ID + 1]]
            descs = self.Descriptors[self.idxs[img_ID]:self.idxs[img_ID + 1]]

            # BFMatcher with default params
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(Q_Descriptors, descs, k=2)

            # Need to draw only good matches, so create a mask
            matchesMask = [False for i in range(len(matches))]

            # SNN ratio test
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.9 * n.distance:
                    matchesMask[i] = True
            tentatives = [m[0] for i, m in enumerate(matches) if matchesMask[i]]

            th = 4.0
            n_iter = 2000
            cmp_H, cmp_mask, num = verify_pydegensac(Q_kps, kps, tentatives, th, n_iter, is_H)
            # draw_matches(kps1, kps2, tentatives, img1, img2, cmp_H, cmp_mask)
            newScores[img_ID] = num

        newList = sorted(newScores.items(), key=lambda x: -x[1])

        newList = np.array([x[0] for x in newList])
        return newList

    def save(self):
        tree_name = '{}_tree_{}_{}_hamming'.format(self.Dataset.name, self.branchs, self.maximum_height)
        prefix_tree = 'Tree/' + tree_name
        BoFs_name = '{}_BoFs_{}_{}_hamming'.format(self.Dataset.name, self.branchs, self.maximum_height)
        prefix_BoFs = 'Tree/' + BoFs_name
        hamming_name = '{}_bitVec_{}_{}_hamming'.format(self.Dataset.name, self.branchs, self.maximum_height)
        prefix_hamming = 'Tree/' + hamming_name
        des2Im_name = '{}_des2Im_{}_{}_hamming'.format(self.Dataset.name, self.branchs, self.maximum_height)
        prefix_des2Im = 'Tree/' + des2Im_name
        with open(prefix_tree, 'wb') as file:
            pickle.dump(self.Tree, file)
        with open(prefix_BoFs, 'wb') as file:
            pickle.dump(self.BoFs, file)
        with open(prefix_hamming, 'wb') as file:
            pickle.dump(self.bit_vec, file)
        with open(prefix_des2Im, 'wb') as file:
            pickle.dump(self.Des_to_Im, file)

