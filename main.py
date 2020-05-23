from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split
import NeuralNetworks

raw_data = pd.read_csv('total.csv')
# print(raw_data)
sentiment = raw_data['setiment_words'].values
y = raw_data['polarity'].values
# print(sentiment)
raw_x = [[word] for word in sentiment]
# word2vec ==> to vector
model = Word2Vec(min_count=1, window=20, size=100, sample=1e-5, workers=4)
model.build_vocab(raw_x)
model.save('sentiment.w2v')
model = Word2Vec.load('sentiment.w2v')
# get word vector
x = np.zeros((len(raw_x), 100))
for i in range(len(raw_x)):
    x[i] = model[raw_x[i]]
print(x.shape, y.shape)
print(np.unique(y))
# training set and test set split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y)

# logistic regression
lr = LogisticRegression(C=1.0, solver='sag', max_iter=400, n_jobs=-1)
lr.fit(x_train, y_train)
print('Logistic regression accuracy: ', lr.score(x_test, y_test))
y_pred = lr.predict(x_test)
ac = np.sum([1 if y_pred[i] == y_test[i] else 0 for i in range(len(y_pred))]) / len(y_pred)

# svm
sigma = 0.1  # sigma影响聚集程度，从而影响泛化程度, sigma小则高斯分布基本作用于支持向量
             # 附近，sigma大则可作用范围变大，泛化程度提高. sigma过小会造成过拟合
gamma = np.power(sigma, -2.) / 2.
svm_model = SVC(C=1.0, kernel='rbf', gamma=gamma)
svm_model.fit(x_train, y_train)
print('svm accuracy: ', svm_model.score(x_test, y_test))


# neural networks
# 先将y转化为one-hot形式, 即(len(y), np.unique(y).shape]) ==> (30804, 3)
def expand(y, kind):
    res = []
    for y_i in y:
        y_array = np.zeros(kind)
        y_array[y_i + 1] = 1
        res.append(y_array)
    return np.array(res)


y_train_softmax = expand(y_train, 3)
print(y_train_softmax.shape)
# theta1(25, 101)  theta2(3, 26)
final_theta = NeuralNetworks.neural_network(x_train, y_train_softmax)
print('neural network accuracy: ',
      NeuralNetworks.accuracy(final_theta, x_test, y_test))
