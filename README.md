### 情感词极性分类器

- 数据集基于total.csv，源自参加南开大学NLP实验室科研活动的时候整理的词语极性数据
- 情感极性分类有 -1(否定) 0(中性) 1(肯定)
- 本项目将对该数据集进行学习

### 项目结构

- main.py: 主执行函数，进行三种不同模型的学习，包括:
  - Logistic Regression
  - SVM
  - Neural Network
  - Native Bayes(未完成)
  - 以后会增加更多模型
- NeuralNetworks.py: 自己写的神经网络模型学习模型
- NativeBayes.py: 自己写的朴素贝叶斯学习模型

#### 词向量转化使用了word2vec,请注意安装依赖！

