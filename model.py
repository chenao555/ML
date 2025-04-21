import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation

        # 改进的权重初始化
        self.params = self._initialize_params()

    def _initialize_params(self):
        layers = [self.input_size] + self.hidden_sizes + [self.output_size]
        params = {}

        for i in range(len(layers) - 1):
            # He初始化
            scale = np.sqrt(2.0 / layers[i])
            params[f'W{i + 1}'] = np.random.randn(layers[i], layers[i + 1]) * scale
            params[f'b{i + 1}'] = np.zeros((1, layers[i + 1]))

        return params

    def _activation(self, x, derivative=False):
        if self.activation == 'relu':
            return np.maximum(0, x) if not derivative else (x > 0).astype(float)
        elif self.activation == 'leaky_relu':
            return np.where(x > 0, x, 0.01 * x) if not derivative else np.where(x > 0, 1.0, 0.01)
        elif self.activation == 'tanh':
            return np.tanh(x) if not derivative else 1 - np.tanh(x) ** 2

    def forward(self, X):
        self.cache = {'A0': X}

        # 向量化前向传播
        for i in range(len(self.hidden_sizes) + 1):
            W = self.params[f'W{i + 1}']
            b = self.params[f'b{i + 1}']

            Z = np.dot(self.cache[f'A{i}'], W) + b

            # 最后一层使用softmax
            A = (self._softmax(Z) if i == len(self.hidden_sizes)
                 else self._activation(Z))

            self.cache[f'Z{i + 1}'] = Z
            self.cache[f'A{i + 1}'] = A

        return self.cache[f'A{len(self.hidden_sizes) + 1}']

    def _softmax(self, X):
        # 数值稳定性softmax
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)

    def compute_loss(self, X, y, reg_strength=0.01):
        m = X.shape[0]
        y_pred = self.forward(X)

        # 交叉熵损失
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        cross_entropy_loss = -np.sum(y * np.log(y_pred)) / m

        # L2正则化
        reg_loss = sum(np.sum(np.square(w)) for w in self.params.values() if 'W' in w)

        return cross_entropy_loss + 0.5 * reg_strength * reg_loss

    def backward(self, X, y, reg_strength=0.01):
        m = X.shape[0]
        self.forward(X)

        # 输出层梯度
        dA_L = (self.cache[f'A{len(self.hidden_sizes) + 1}'] - y) / m
        grads = {}

        # 反向传播
        for i in range(len(self.hidden_sizes) + 1, 0, -1):
            W = self.params[f'W{i}']
            A_prev = self.cache[f'A{i - 1}']

            # 梯度计算
            if i > 1:
                dZ = dA_L * self._activation(self.cache[f'Z{i}'], derivative=True)
            else:
                dZ = dA_L

            dW = np.dot(A_prev.T, dZ) + reg_strength * W
            db = np.sum(dZ, axis=0, keepdims=True)

            # 前一层梯度
            dA_L = np.dot(dZ, W.T)

            grads[f'dW{i}'] = dW
            grads[f'db{i}'] = db

        return grads

    def predict(self, X):
        # 直接返回前向传播的概率分布
        probabilities = self.forward(X)
        return probabilities