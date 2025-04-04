import pandas as pd
import numpy as np
def pca(data,k):
    # 中心化
    data_mean = np.mean(data,axis = 0)# 指定按列平均
    for i in range(data.shape[1]):
        data[:,i] = data[:,i] - data_mean[i]
    # print(data)

    # 求协方差矩阵
    data_cov = np.dot(data.T,data) / (data.shape[0]-1)
    # print(data_cov.shape)

    # 求特征值和特征向量
    data_values, data_vectors = np.linalg.eig(data_cov)
    sorted_indices = np.argsort(data_values)[::-1]  # 从大到小排序
    data_values = data_values[sorted_indices]
    data_vectors = data_vectors[:, sorted_indices]
    # print(data_values.shape,data_vectors.shape)

    # 选择前k个特征向量
    top_vectors = data_vectors[:, :k]

    # 进行PCA降维
    data_pca = np.dot(data, top_vectors)

    print("原数据：{}".format(data.shape))
    print("中心化数据：{}".format(data.shape))
    print("协方差矩阵：{}".format(data_cov.shape))
    print("特征向量：{}".format(data_vectors.shape))
    print("PCA降维后的数据：{}".format(data_pca.shape))
    return data_pca


