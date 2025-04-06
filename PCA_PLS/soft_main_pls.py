import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
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

# 以vis数据为例 pca降维+pls回归
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
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(vis_train_x, vis_train_y, test_size=0.2, random_state=42)
 
# 创建并训练PLS回归模型，假设提取2个主成分
pls = PLSRegression(n_components=2)
pls.fit(X_train, y_train)
 
# 使用训练好的模型进行预测
y_pred = pls.predict(X_test)
print("测试值大小{}".format(X_test.shape))
print("预测值大小{}".format(y_pred.shape))

# 计算预测结果的R²得分（决定系数）
r2 = r2_score(y_test, y_pred)
print("模型在测试集上的R²得分：{}".format(r2))
 
# 画图
plt.figure(figsize=(12,6), dpi=150)
plt.plot([x for x in range(len(y_test))],y_test,marker='o', label='vis_real')
plt.plot([x for x in range(len(y_pred))],y_pred,marker='x', label='vis_predict')
plt.text(5, 5, 'R^2: {}%'.format(r2*100), fontsize=12, color='red')
plt.title('PCA/PLS, R2: {}%'.format(r2*100))
plt.legend()
plt.show()
