import numpy as np


# 朴素贝叶斯，假设每个特征互相独立
def train_bayes(x, y):
    m = len(x)
    features = x.shape[1]
    num_classes = []
    y_unique = np.unique(y)
    for y_i in y_unique:
        num_yi = np.sum([1 if y_i == y[i] else 0 for i in range(m)])
        num_classes.append(num_yi)
    p_class = np.array(num_classes) / m  # P(y == k)
    p_num_class = np.array([np.ones(features) for i in y_unique])
    p_num_denom = [2.0 for i in y_unique]

    for label_index in range(len(y_unique)):
        for line in range(m):
            if y[line] == y_unique[label_index]:
                p_num_class[label_index] += x[line]  # 每个特征在样本数的总和
                p_num_denom[label_index] += np.sum(x[line])

    p_class_vect = []
    # 求P(xi| y == k) 其中xi为样本特征, k为对应的标签值
    # 在这次的训练中，这次的过程是求P(xi | y == -1) P(xi | y == 0) P(xi | y == 1)
    for label_index in range(len(y_unique)):
        # 使用log防止下溢
        p_class_vect.append(
            np.log(p_num_class[label_index] / p_num_denom[label_index]))

    # log(P(xi| y == k))  P(y == k)
    return np.array(p_class_vect), p_class


# 利用所得模型参数进行预测
# 比较时无需求出P(y == k | x)。由P(y == k | x) = P(x | y == k) * P(y == k) / P(x)可知
# 因为每个分类的P(x)相同，因此只需要比较P(x | y == k) * P(y == k)即可，哪个值最大就分到
# 哪个类(一共k类)  用对数将连乘改为求和，并减少上溢的可能
def predict_bayes(p_class_vect, p_class, x, y):
    y_pred = []
    y_unique = np.unique(y)
    for i in range(len(x)):
        p_predict = []
        for label_index in range(len(y_unique)):
            p_class_i = np.sum(p_class_vect[label_index] * x[i]) + \
                        np.log(p_class[label_index])
            p_predict.append(p_class_i)
        # 比较，并选择概率最大的标签
        max_pro, max_j = -1e6, -1
        for j in range(len(y_unique)):
            if max_pro < p_predict[j]:
                max_pro = p_predict[j]
                max_j = j
        y_pred.append(max_j)
    return np.array(y_pred)


# 预测
def accuracy(y_pred, y):
    return np.sum([1 if y_pred[i] == y[i]
                   else 0 for i in range(y_pred.shape[0])]) / y_pred.shape[0]
