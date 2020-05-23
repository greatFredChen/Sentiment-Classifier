import numpy as np


# 使用one-hot编码
def one_hot(raw_x):
    # 建立词典
    word_dict, index = {}, 0
    for word in raw_x:
        if not word_dict.__contains__(word):
            word_dict[word] = index
            index = index + 1
    # 由词典生成词语向量
    word_vec = np.zeros((raw_x.shape[0], index))
    for i in range(raw_x.shape[0]):
        w_index = word_dict[raw_x[i]]
        word_vec[i][w_index] = 1.
    return word_vec
