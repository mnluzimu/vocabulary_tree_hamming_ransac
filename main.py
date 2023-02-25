import Dataset
import VocTree
import cv2
from utils import compute_map_and_print
import numpy as np
import pickle as pkl
import os

if __name__ == '__main__':
    root_path = r"F:\\science_research\\image_retrieval\\dataset\\oxford5k\\oxbuild_images-v1"
    sift_path = r"./oxford5k/oxford5k_sift"
    projection_path = r"./oxford5k/oxford5k_projection"
    kp_path = r"./oxford5k/oxford5k_kp"
    gt_fn = r".\oxford5k\gt.pkl"
    gnt = pkl.load(open(gt_fn, 'rb'))
    is_H = False
    Hamming = False
    Ransac = True
    n_rerank = 300


    dataset = Dataset.Oxford5k(root_path=root_path,
                               sift_path=sift_path,
                               projection_path=projection_path,
                               kp_path=kp_path)
    vocTree = VocTree.VocTree(Dataset=dataset,
                              Tree_path=r"./Tree/Oxford5k_tree_10_6_hamming",
                              BoFs_path=r"./Tree/Oxford5k_BoFs_10_6_hamming",
                              bitVec_path=r"./Tree/Oxford5k_bitVec_10_6_hamming",
                              des2Im_path=r"./Tree/Oxford5k_des2Im_10_6_hamming",
                              Train=True,
                              branchs=3,
                              maximum_height=2)
    Q_path = "./oxford5k/oxford5k_sift/all_souls_000002.jpg.sift"
    Q_path_pro = "./oxford5k/oxford5k_projection/all_souls_000002.jpg.proj"
    img = cv2.imread("./oxford5k/oxbuild_images-v1/all_souls_000002.jpg")

    ranks = []
    for k, data in enumerate(gnt):
        print('>>\r{}/55'.format(k + 1), end='')
        Q_ID = data['query']
        img_fn = os.path.join(root_path, dataset.Image_names[Q_ID])
        query_fn = os.path.join(sift_path, dataset.Image_names[Q_ID] + '.sift')
        pro_fn = os.path.join(projection_path, dataset.Image_names[Q_ID] + '.proj')

        if Hamming:
            rank = vocTree.Query_with_Hamming(Q_image_ID=Q_ID,
                                              root_node=vocTree.Tree,
                                              ht=100,
                                              result_size=10000)
        else:
            rank = vocTree.Query(Q_image_ID=Q_ID,
                                 root_node=vocTree.Tree,
                                 BoFs=vocTree.BoFs,
                                 result_size=10000)

        if Ransac:
            rerank_list = vocTree.reRank(rank[:n_rerank], Q_ID, is_H)
            rank[:n_rerank] = rerank_list

        rank = rank.reshape(-1, 1)
        ranks.append(rank)

    print('')

    ranks = np.concatenate(ranks, axis=1)

    compute_map_and_print('roxford5k', ranks, gnt)


    # rank = vocTree.Query_with_Hamming(2, root_node=vocTree.Tree,
    #                                  ht=10,
    #                                  result_size=10)
    # rank = vocTree.Query(Q_image_ID=2, root_node=vocTree.Tree, BoFs=vocTree.BoFs, result_size=10000)
    # # rank = vocTree.Query_with_Hamming(Q_image_ID=2,
    # #                                root_node=vocTree.Tree,
    # #                                ht=10,
    # #                                result_size=10000)
    # rank = vocTree.reRank(rank[:n_rerank], 2, is_H)
    # q_img = cv2.imread(os.path.join(root_path, dataset.Image_names[2]))
    # cv2.imshow("q_image", q_img)
    # for k, v in enumerate(rank[:50]):
    #     print("rank {}: {}".format(k, dataset.Image_names[v]))
    #     img = cv2.imread(os.path.join(root_path, dataset.Image_names[v]))
    #     cv2.imshow("rank {}: {}".format(k, dataset.Image_names[v]), img)
    #
    #
    # cv2.waitKey(0)
