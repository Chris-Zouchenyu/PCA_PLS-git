import pandas as pd
import numpy as np
from pca import pca
import matplotlib.pyplot as plt
import seaborn as sns
# 设置字体
plt.rcParams['font.sans-serif'] = 'Times New Roman'  
plt.rcParams['axes.unicode_minus'] = False 
##################################### 对硫回收数据进行可视化 #####################################
def plot_2(data):
    '''
    创建带回归线的散点图
    '''
    sns.regplot(x='X', y='Y', data=data, color='blue', label='Sulfur recovery data 2 dimmension')  # 绘制散点图和回归线
    plt.title('DataPlot_2')  # 添加标题
    plt.xlabel('X Axis')  # 添加 X 轴标签
    plt.ylabel('Y Axis')  # 添加 Y 轴标签
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    plt.show()  # 显示图表
def plot_3(data):
    '''
    创建带回归线的散点图
    创建三维图形'
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # 添加三维子图
    ax.scatter(data[:,0], data[:,1], data[:,2], c='r', marker='o', label='Sulfur recovery data 3 dimmension')# 绘制散点图
    ax.set_title('DataPlot_3')# 设置标题和标签
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.legend()# 显示图例
    plt.show()# 显示图形

if __name__ == '__main__':
    data = pd.read_csv(r'data\硫回收单元数据SRU_data.txt',header = None, index_col= None, sep='\s+') # 不指定行名列名
    data = np.array(data, dtype = np.float32)# 转换为浮点值
    # print(data.shape)# (10081, 7)
    k = 3
    data_pca_3 = pca(data,k)# (10081, 3)
    k = 2
    data_pca_2 = pca(data,k)# (10081, 2)
    # 可视化 2维
    data_plot_2 = pd.DataFrame({
        'X': data_pca_2[:,0],
        'Y': data_pca_2[:,1]
    })
    plot_2(data_plot_2)
    # 可视化 3维
    plot_3(data_pca_3)



