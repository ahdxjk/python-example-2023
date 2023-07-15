import collections
import math
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import os


def preprocessing(x_train, y_train):
    # x = x_train[:, :-1]
    # y = x_train[:, -1]
    x = x_train
    y = y_train
    max_value = np.max(x[:, :-1])
    min_value = np.min(x[:, :-1])
    # print(min_value, max_value)
    # for i in range(len(x)):
    norm_value = (x - min_value) / (max_value - min_value)
    # print(norm_value)
    # 对数据进行升维
    data_sum = np.square(norm_value)
    # print(data_sum)
    square_sum = data_sum.sum(axis=1)
    # print(square_sum)
    # 升维
    R = np.max(square_sum)
    # print(R)
    values = np.sqrt(np.square(R) - np.square(square_sum))
    # print(values)
    x = np.c_[norm_value, values]
    # print(x)
    # x = np.insert(x, x.shape[1]+1, [0], axis=1)
    # print(data)
    data = np.c_[x, y]
    # print(data)
    return data


# 训练函数
def fit(data):
    # train_data = data[:, :-1]
    # train_lab = data[:, -1]
    # cover = np.zeros(len(x))
    cover_points = []         # 保存覆盖样本点信息
    # Cover = {'center': [], 'radius': [], 'class': []}
    while len(data) != 0:
        if data.size == 1:      # 如果训练数据中只有一个样本点
            Cover = {'center': [], 'radius': [], 'class': []}
            feature_data = data[0][:-1]
            r = np.inner(feature_data, feature_data)        # 计算与自己的内积作为半径
            Cover['center'].append(feature_data)
            Cover['radius'].append(r)
            Cover['class'].append(data[0][-1])
            cover_points.append(Cover)
            data = np.delete(data, 0, axis=0)               # 将已经覆盖的样本点删除
        else:
            # 随机选取一个样本点
            # s = np.random.choice(train)
            # s_x = train[s, :-1]
            index_list = [i for i in range(len(data))]      # 为每个数据建立一个索引号，长度等于剩余的数据长度
            index = random.choice(index_list)               # 随机选取一个索引
            s = data[index]                                 # 选择的数据
            cover_index = []                                # 用来保存一个覆盖内的索引
            cover_index.append(index)                       # 覆盖中心的索引
            lab = s[-1]
            # 初始化异类点内积d1,同类点内积d2
            d1 = 0
            d2 = np.inner(s[:-1], s[:-1])
            count = 0                                       # 用来找第一个异类点以及判断是否含有异类点
            for temp_data in data:
                if temp_data[-1] != lab:                    # 存在异类点
                    count = count + 1                       # 异类点个数加1
                    if count == 1:                          # 只有一个异类点
                        d1 = np.inner(s[:-1], temp_data[:-1])    # 计算与该异类点之间的内积
                    elif count > 1:
                        temp_d = np.inner(s[:-1], temp_data[:-1])
                        if temp_d > d1:
                            d1 = temp_d

            if d1 == 0:  # 只有同类
                Cover = {'center': [], 'radius': [], 'class': []}
                feature_data = s[:-1]
                r = np.inner(feature_data, feature_data)    # 把同类点之间的内积作为覆盖半径
                Cover['center'].append(feature_data)
                Cover['radius'].append(r)
                Cover['class'].append(s[-1])
                data = np.delete(data, index, axis=0)      # 将单覆盖删除
                cover_points.append(Cover)
            else:  # 有异类点，需要在最近异类点d1内找同类点
                for temp_s in data:
                    if temp_s[-1] == s[-1]:           # 同类点
                        temp_same_d = np.inner(temp_s[:-1], s[:-1])        # 计算同类点之间的内积
                        if temp_same_d > d1:
                            if d2 > temp_same_d:
                                d2 = temp_same_d

                if d2 == np.inner(s[:-1], s[:-1]):  # 有最近异类，在此距离内找不到最远同类（噪音点）
                    Cover = {'center': [], 'radius': [], 'class': []}
                    feature_data = s[:-1]
                    r = np.inner(feature_data, feature_data)          # 将与自己的内积作为覆盖半径
                    Cover['center'].append(feature_data)
                    Cover['radius'].append(r)
                    Cover['class'].append(s[-1])
                    cover_points.append(Cover)
                    data = np.delete(data, index, axis=0)  # 将单覆盖删除
                else:  # 有最近异类，在此距离内有最远同类
                    Cover = {'center': [], 'radius': [], 'class': []}
                    r = (d1 + d2) / 2         # 折中半径法
                    Cover['center'].append(s[:-1])
                    Cover['radius'].append(r)
                    Cover['class'].append(s[-1])
                    cover_points.append(Cover)
                    for i in range(len(data)):
                        if data[i][-1] == s[-1]:
                            temp_same_d = np.inner(data[i][:-1], s[:-1])      # 找最远同类点
                            if r < temp_same_d:
                                cover_index.append(i)
                    data = np.delete(data, cover_index, axis=0)         # 将在覆盖区域内的样本点删除
    # print(cover_points)
    return cover_points


# 测试函数
def CCA_test(x, feature, cover_points):         # 对测试数据进行升维
    predict_data = []
    true_data = []
    x_test = feature

  # 取测试数据以及对应的标签
    x_test_predict = []  # 保存预测标签
    for temp_test in x_test:
        test_nearest_distance = []  # 保存距离边界最近的样本点内积
        temp_cover_class = []  # 保存最近距离所对应的类别
        for temp_cover in cover_points:
            temp_d = np.inner(temp_test[:], temp_cover['center'][0][:])  # 计算与覆盖中心的内积
            test_nearest_distance.append(temp_d - temp_cover['radius'])  # 靠近边界距离
            temp_cover_class.append(temp_cover['class'])  # 找出覆盖的样本点的类别
        # 先找出最近的距离，在找出对应的索引以及对应的类别
        x_test_predict.append(temp_cover_class[test_nearest_distance.index(max(test_nearest_distance))])
    correct_num = 0  # 初始化预测正确样本的个数
    #print(predict_data)
    #print(true_data)
    return x_test_predict




if __name__ == '__main__':
    pred1 =[]
    true1 = []
    dataPath = r'C:\Users\27689\Desktop\feature.csv'
    data = pd.read_csv(dataPath, header = None).to_numpy()  # 找出每个文件对应的路径名，并读取相应的数据集，将其格式转换为numpy形式
    # x = data[:, :-1]
    # y = data[:, -1]
    x = data[: , : -1]
    y = data[:, -1]
    hhh = fit(data)
    feature =x[ :1, : ]
    a = CCA_test(x, feature, hhh)
    print(a)










