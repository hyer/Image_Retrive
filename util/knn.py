# coding=UTF-8

import numpy as np


class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        self.X_tr = X
        self.y_tr = y

    def predict(self, X, k=1, to_out_score=True):

        num_train = self.X_tr.shape[0]

        if k > num_train:
            raise ValueError("K cannot be less then num of Train!")

        num_test = X.shape[0]
        y_pred = np.zeros(num_test, dtype=self.y_tr.dtype)
        y_score = np.zeros(num_test, dtype='float')

        for i in xrange(num_test):

            num_train = self.X_tr.shape[0]

            '''
            # unvectorized version 计算测试样本，与所有训练样本的余弦相似性

            sim_label_list = []

            for j in xrange(num_train):
                sim = cosine_similarity(X[i, :], self.X_tr[j, :])
                label = self.y_tr[j]
                sim_label_list.append((sim, label))
            '''

            # vectorized version 更加快速  cosine_similarity
            A = np.tile(X[i, :], (num_train, 1))
            B = self.X_tr

            tmp1 = np.sum(A * B, axis=1)
            tmp2 = (np.sum(A ** 2, axis=1) ** 0.5) * (np.sum(B ** 2, axis=1) ** 0.5)
            similarity = tmp1 / (tmp2 + 1e-12)

            sim_list = similarity.tolist()
            sim_label_list = zip(sim_list, self.y_tr)

            # 按照相似性得分进行排序
            sim_label_list_sorted = sorted(sim_label_list, key=lambda x: x[0], reverse=True)

            # k近邻计数
            dic = {}
            for sim, label in sim_label_list_sorted[:k]:
                if label not in dic:
                    dic[label] = [1, sim]
                else:
                    dic[label][0] += 1
                    if sim > dic[label][1]:
                        dic[label][1] = sim

            # 根据计数结果进行排序，如果计数结果相同，则根据相似性进行排序
            sorted_list = sorted(dic.items(), key=lambda d: (d[1][0], d[1][1]), reverse=True)

            label, (count, sim) = sorted_list[0]
            y_pred[i] = label
            y_score[i] = sim

            #vectorized version 更加快速  l2_dist
            # A = np.tile(X[i, :], (num_train, 1))
            # B = self.X_tr
            #
            # diff_mat = A - B
            # sq_diff_mat = diff_mat ** 2  # 差值矩阵平方
            # sq_distances = sq_diff_mat.sum(axis=1)  # 计算每一行上元素的和
            # dist_list = sq_distances.tolist()
            # dist_label_list = zip(dist_list, self.y_tr)
            #
            #
            # # 按照距离进行排序
            # dist_label_list_sorted = sorted(dist_label_list, key=lambda x: x[0])
            #
            # # k近邻计数
            # dic = {}
            # for dist, label in dist_label_list_sorted[:k]:
            #     if label not in dic:
            #         dic[label] = [1, dist]
            #     else:
            #         dic[label][0] += 1
            #         if dist < dic[label][1]:
            #             dic[label][1] = dist
            #
            # # 根据计数结果进行排序，如果计数结果相同，则根据相似性进行排序
            # sorted_list = sorted(dic.items(), key=lambda d: (-d[1][0], d[1][1]))
            #
            # label, (count, dist) = sorted_list[0]
            # y_pred[i] = label
            # y_score[i] = dist

        if to_out_score:
            return y_pred, y_score
        else:
            return y_pred