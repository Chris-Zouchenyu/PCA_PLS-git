from scipy.io import loadmat
from pca import pca
from model.BP import BP_model
from model.CNN import CNN_model
import numpy as np
import os,torch
import torchkeras
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import train_test_split

def train(train_x, train_y,k,module,mode):
    '''
    训练函数
    train_x : x
    train_y : y
    k : 维度
    '''
    if mode == 'train':
        # pca降维(自己写的函数)
        train_x = pca(train_x,k)# (252, 401) -> (252,20)

        # 训练集 测试集的划分
        X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

        # 开始训练
        if module == 'BP':   
            model = torchkeras.Model(BP_model(k))
        elif module == 'CNN':
            X_train,y_train = split_windows(X_train,y_train,7)
            X_train = torch.tensor(X_train, dtype = torch.float32)
            y_train = torch.tensor(y_train, dtype = torch.float32)
            X_train = X_train.unsqueeze(1)
            # print(train_x.shape)
            model = torchkeras.Model(CNN_model(k))
        train_x = torch.tensor(X_train, dtype = torch.float32)
        train_y = torch.tensor(y_train, dtype = torch.float32)
        ds_train = TensorDataset(train_x, train_y)# 数据加载
        dl_train = DataLoader(ds_train, batch_size=4, num_workers=0)
        
        # 超参数
        lr = 0.001
        EPOCH = 300
        loss_fn = torch.nn.MSELoss()# 损失函数
        optimizer = torch.optim.Adam(model.parameters(),lr=lr)# 优化器
        # torchkeras训练
        model.compile(loss_func=loss_fn, optimizer=optimizer)
        model.fit(epochs=EPOCH,dl_train=dl_train)
        if module == 'BP':
            torch.save(model,r'parameters\lr1e^-3_epoch30.pth')
        if module == 'CNN':
            torch.save(model,r'parameters\CNN_lr1e^-3_epoch30.pth')
        return X_train, y_train
    else:
        print('未进行训练')


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

# 数据导入
file_path = 'data\柴油质量指标光谱分析数据'
file_list = []
mat = []
file_names = os.listdir(file_path)
for i in file_names:
    file_name = file_path + '\\' + i
    file_list.append(file_name)
    mat_data = loadmat(file_name)
    mat_data_key = np.array(list(mat_data.keys()))

# 以vis数据为例 pca降维+bp回归
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
# train(vis_train_x,vis_train_y,k = 20,module='BP',mode='train')

# pca降维+CNN回归
train(vis_train_x,vis_train_y,k = 20,module='CNN',mode='train')



