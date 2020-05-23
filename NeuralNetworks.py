import numpy as np
import scipy.optimize as opt


# 矩阵权重随机初始化
# theta1(25, 101)  theta2(3, 26)
def random_init(shape1, shape2):
    theta1 = np.random.uniform(-0.12, 0.12, shape1)
    theta2 = np.random.uniform(-0.12, 0.12, shape2)
    return theta1, theta2


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def gradient_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


# 前向传播
def forward(theta, x):
    t1, t2 = deserialize(theta)
    a1 = np.insert(x, 0, 1, axis=1)  # (x, 101)
    z2 = a1 @ t1.T  # (x, 25)
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, 1, axis=1)  # (x, 26)
    z3 = a2 @ t2.T  # (x, 3)
    a3 = sigmoid(z3)
    return a1, z2, a2, z3, a3


def cost(theta, x, y):
    a1, z2, a2, z3, hx = forward(theta, x)
    m = len(x)
    first = -y * np.log(hx)
    second = (1 - y) * np.log(1 - hx)
    return np.sum(first - second) / m


def regularized_cost(theta, x, y, l=1.):
    Cost = cost(theta, x, y)
    m = len(x)
    # 正则项不需要惩罚bias
    t1, t2 = deserialize(theta)
    reg = np.sum(t1[:, 1:] ** 2) + \
          np.sum(t2[:, 1:] ** 2)
    return Cost + l * reg / (2 * m)


def gradient(theta, x, y):
    t1, t2 = deserialize(theta)
    a1, z2, a2, z3, hx = forward(theta, x)
    d3 = hx - y  # (x, 3)
    # (x, 3) @ (3, 25) * (x, 25) = (x, 25)
    d2 = d3 @ t2[:, 1:] * gradient_sigmoid(z2)  # (x, 25)
    D2 = d3.T @ a2  # (3, x) @ (x, 26) = (3, 26)
    D1 = d2.T @ a1  # (25, x) @ (x, 101) = (25, 101)
    return (1. / len(x)) * serialize(D1, D2)


def regularized_gradient(theta, x, y, l=1.):
    D = gradient(theta, x, y)
    D1, D2 = deserialize(D)  # 1 / m * Di
    t1, t2 = deserialize(theta)
    # 不惩罚bias项
    t1[:, 0] = 0
    t2[:, 0] = 0
    regD1 = D1 + (l / len(x)) * t1
    regD2 = D2 + (l / len(x)) * t2
    return serialize(regD1, regD2)


def serialize(t1, t2):
    theta = np.r_[t1.flatten(),
                  t2.flatten()]
    return theta


def deserialize(theta):
    return theta[: 25 * 101].reshape(25, 101), theta[25 * 101:].reshape(3, 26)


def neural_network(x, y, l=1.):
    theta1, theta2 = random_init((25, 101), (3, 26))  # theta1(25, 101)  theta2(3, 26)
    init_theta = serialize(theta1, theta2)
    res = opt.minimize(fun=regularized_cost,
                       x0=init_theta,
                       args=(x, y, l),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'maxiter': 450})
    return res.x


def accuracy(theta, x, y):
    a1, z2, a2, z3, hx = forward(theta, x)  # hx (test, 3)
    y_pred = np.argmax(hx, axis=1) - 1
    return np.sum([1 if y_pred[i] == y[i] else 0 for i in range(len(x))]) / len(x)
