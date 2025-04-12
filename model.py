from utils import *

class MLP3:
    def __init__(self, input_dim=3072, hidden_dim=512, output_dim=10,
                 activation='relu'):
        """
        三层神经网络模型 (输入层 → 隐藏层 → 输出层)
        参数:
            input_dim: 输入维度 (CIFAR-10: 32x32x3=3072)
            hidden_dim: 隐藏层神经元数量
            output_dim: 输出类别数 (CIFAR-10:10)
            activation: 激活函数 ['relu', 'sigmoid', 'tanh']
            weight_scale: 权重初始化策略 ['auto', float数值]
        """
        # 初始化参数
        self.params = {}
        self.activation = activation.lower()

        # 权重初始化（标准正态分布初始化）
        self.params['W1'] = np.random.randn(input_dim, hidden_dim) * 0.01
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.randn(hidden_dim, output_dim) * 0.01
        self.params['b2'] = np.zeros(output_dim)

        # 选择激活函数
        self._init_activation(activation)

    def _init_activation(self, activation):
        """初始化激活函数及其导数"""
        activation = activation.lower()
        if activation == 'relu':
            self.activation_func = lambda x: np.maximum(0, x)
            self.activation_deriv = lambda x: (x > 0).astype(np.float32)
        elif activation == 'sigmoid':
            self.activation_func = lambda x: 1 / (1 + np.exp(-x))
            self.activation_deriv = lambda x: self.activation_func(x) * (
                        1 - self.activation_func(x))
        elif activation == 'tanh':
            self.activation_func = lambda x: np.tanh(x)
            self.activation_deriv = lambda x: 1 - np.tanh(x) ** 2
        else:
            raise ValueError(f"不支持的激活函数: {activation}")


    def forward(self, X):
        """
        前向传播
        X: 输入数据 (batch_size, input_dim)
        y_pred: 预测概率 (batch_size, output_dim)
        cache: 缓存中间结果用于反向传播

        """
        # 输入验证
        assert X.shape[1] == self.params['W1'].shape[0], \
            f"输入维度{X.shape[1]}与W1的输入维度{self.params['W1'].shape[0]}不匹配"

        # 第一层计算
        Z1 = np.dot(X, self.params['W1']) + self.params['b1']
        A1 = self.activation_func(Z1)
        dA1 = self.activation_deriv(Z1)  # 激活函数导数

        # 输出层计算
        Z2 = np.dot(A1, self.params['W2']) + self.params['b2']
        y_pred = softmax(Z2)

        # 缓存中间结果
        cache = (X, Z1, A1, dA1)
        return y_pred, cache

    def backward(self, y_pred, y_true, cache, lambda_reg=0.0):
        """
        反向传播 (含L2正则化)
        y_pred: 模型预测概率 (batch_size, output_dim)
        y_true: 真实标签的one-hot编码 (batch_size, output_dim)
        cache: 前向传播的缓存结果
        lambda_reg: L2正则化强度系数
        grads: 包含各参数梯度的字典
        """
        X, Z1, A1, dA1 = cache
        batch_size = y_true.shape[0]

        # 输出层梯度
        delta2 = (y_pred - y_true) / batch_size  # 交叉熵梯度
        grads = {}
        grads['W2'] = np.dot(A1.T, delta2) + lambda_reg * self.params[
            'W2']  # L2正则化
        grads['b2'] = np.sum(delta2, axis=0)

        # 隐藏层梯度
        delta1 = np.dot(delta2, self.params['W2'].T) * dA1
        grads['W1'] = np.dot(X.T, delta1) + lambda_reg * self.params[
            'W1']  # L2正则化
        grads['b1'] = np.sum(delta1, axis=0)

        return grads


if __name__ == '__main__':
    model  = MLP3()
    X = np.random.randn(5000, 3072)  # 标准正态分布
    y,_ = model.forward(X)
    print(f"X的形状：{X.shape}\ny的形状：{y.shape}")
    for i in range(10):
        print(y[i],"\n")
    print(model.params)

