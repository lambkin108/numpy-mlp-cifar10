import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(np.float32)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def softmax(Z):
    shift_Z = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(shift_Z)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def to_one_hot(y, num_classes=10):
    """ 将类别转换为独热编码 """
    return np.eye(num_classes)[y]

def accuracy(y_pred, y_true):
    """ 计算分类准确率 """
    # 将预测概率转换为类别索引
    pred_labels = np.argmax(y_pred, axis=1)
    # 将one-hot标签转换为类别索引
    true_labels = np.argmax(y_true, axis=1)
    # 计算正确预测的数量
    correct = np.sum(pred_labels == true_labels)
    # 计算准确率
    return correct / y_true.shape[0]
