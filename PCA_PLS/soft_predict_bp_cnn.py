from scipy.io import loadmat
from pca import pca
import numpy as np
import matplotlib.pyplot as plt
import os,torch
from sklearn.model_selection import train_test_split
def R_squared(y_true,y_pred):
    '''
    计算r2
    '''
    ss_total = np.sum((y_true-np.mean(y_true))**2)
    ss_residual = np.sum((y_true-y_pred)**2)
    r2 = 1-(ss_residual/ss_total)
    return r2
def split_windows(train_x,train_y, size):
    '''
    划分窗口 窗口为7
    '''
    X = []
    Y = []
    # X作为数据，Y作为标签
    # 滑动窗口，步长为1，构造窗口化数据，每一个窗口的数据标签是窗口末端的close值（收盘价格）
    for i in range(len(train_x) - size):
        X.append(train_x[i:i+size, :])
        Y.append(train_y[i:i+size, :])
    return np.array(X), np.array(Y)
def inverse_split_windows(X, Y):
    '''
    从窗口化的 X 和 Y 反推出原始的 train_x 和 train_y
    '''
    # 获取窗口大小
    size = X.shape[1]
    
    # 初始化 train_x 和 train_y
    train_x = []
    train_y = []
    
    # 添加第一个窗口的数据
    train_x.extend(X[0])
    train_y.extend(Y[0])
    
    # 遍历后续窗口，添加每个窗口的最后一个数据点
    for i in range(1, len(X)):
        train_x.append(X[i][-1])  # 添加当前窗口的最后一个数据点
        train_y.append(Y[i][-1])  # 添加当前窗口的最后一个数据点
    
    # 转换为 NumPy 数组
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    
    return train_x, train_y
# 数据导入 与之前相同
file_path = 'data\柴油质量指标光谱分析数据'
file_list = []
mat = []
file_names = os.listdir(file_path)
for i in file_names:
    file_name = file_path + '\\' + i
    file_list.append(file_name)
    mat_data = loadmat(file_name)
    mat_data_key = np.array(list(mat_data.keys()))

# 以vis数据为例 pca降维+回归
vis_train_x = []
vis_train_y = []
for i in mat_data_key[3:]:
    if i == 'v_sd_hl':
        vis_train_x.append(mat_data[i])
    elif i == 'v_sd_ll_a':
        vis_train_x.append(mat_data[i])
    elif i == 'v_sd_ll_b':
        vis_train_x.append(mat_data[i])
    else:
        vis_train_y.append(mat_data[i])
vis_train_x = np.vstack(vis_train_x)
vis_train_y = np.vstack(vis_train_y)
# print(vis_train_x.shape,vis_train_y.shape)# (252, 401) (252, 1)

# pca降维(自己写的函数)
k = 20
vis_train_x = pca(vis_train_x,k)# (252, 401) -> (252,10)

# 训练集 测试集的划分
X_train, X_test, y_train, y_test = train_test_split(vis_train_x, vis_train_y, test_size=0.2, random_state=42)

module = 'BP'
# 预测部分 BP
if module == 'BP':
    model = torch.load('parameters\lr1e^-3_epoch30.pth')
    vis_train = torch.tensor(X_test,dtype=torch.float32)
    y = model(vis_train)
    with torch.no_grad():
        y = np.array(y)
    r2 = R_squared(y_test,y)
    print('R^2: {}%'.format(r2*100))
    # 画图
    plt.figure(figsize=(12,6), dpi=150)
    plt.plot([x for x in range(len(y_test))],y_test,marker='o', label='vis_real')# 看前100组数据
    plt.plot([x for x in range(len(y))],y,marker='x', label='vis_predict')
    plt.text(5, 5, 'R^2: {}%'.format(r2*100), fontsize=12, color='red')
    plt.title('PCA/PLS, R2: {}%'.format(r2*100))
    plt.legend()
    plt.show()
elif module == 'CNN':
    model = torch.load('parameters\CNN_lr1e^-3_epoch30.pth')
    X_test,y_test = split_windows(X_test,y_test,7)
    X_test = torch.tensor(X_test, dtype = torch.float32)
    X_test = X_test.unsqueeze(1)
    y = model(X_test)
    with torch.no_grad():
        y = np.array(y)
        _, y = inverse_split_windows(X_test,y)
        _, y_test = inverse_split_windows(X_test,y_test)
        r2 = R_squared(y_test,y)
        print('R^2: {}%'.format(r2*100))
        # print(y.shape) #(50,1)
        # 画图
        plt.figure(figsize=(12,6), dpi=150)
        plt.plot([x for x in range(len(y_test))],y_test,marker='o', label='vis_real')# 看前100组数据
        plt.plot([x for x in range(len(y))],y,marker='x', label='vis_predict')
        plt.text(5, 5, 'R^2: {}%'.format(r2*100), fontsize=12, color='red')
        plt.title('PCA/CNN, R2: {}%'.format(r2*100))
        plt.legend()
        plt.show()

# BP 87.91%
# CNN 58.02%