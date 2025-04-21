import numpy as np
import time


class Trainer:
    def __init__(self,
                 model,
                 learning_rate=0.001,  # 默认学习率
                 batch_size=512,  # 默认批量大小
                 max_epochs=50,  # 默认最大训练轮次
                 early_stopping_patience=5  # 早停耐心
                 ):
        # 明确初始化所有关键属性
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs  # 确保有这个属性
        self.early_stopping_patience = early_stopping_patience

        # 训练过程指标存储
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def compute_accuracy(self, X, y):
        """
        计算模型准确率
        """
        predictions = self.model.forward(X)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y, axis=1)
        return np.mean(pred_classes == true_classes)

    def train(self, X_train, y_train, X_val, y_val):
        """
        训练主流程
        """
        m = X_train.shape[0]
        best_val_loss = float('inf')
        patience_counter = 0

        # 计算批次数量
        num_batches = (m + self.batch_size - 1) // self.batch_size

        # 重置指标
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        for epoch in range(self.max_epochs):
            # 数据随机打乱
            shuffle_idx = np.random.permutation(m)
            X_train_shuffled = X_train[shuffle_idx]
            y_train_shuffled = y_train[shuffle_idx]

            epoch_train_loss = 0

            # 批量训练
            for batch in range(num_batches):
                start = batch * self.batch_size
                end = min((batch + 1) * self.batch_size, m)

                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                # 反向传播
                grads = self.model.backward(X_batch, y_batch)

                # 损失计算
                batch_loss = self.model.compute_loss(X_batch, y_batch)
                epoch_train_loss += batch_loss

                # 参数更新 (简化版梯度下降)
                for k in range(1, len(self.model.hidden_sizes) + 2):
                    self.model.params[f'W{k}'] -= self.learning_rate * grads[f'dW{k}']
                    self.model.params[f'b{k}'] -= self.learning_rate * grads[f'db{k}']

                    # 计算整个训练集和验证集的损失和准确率
            train_loss = self.model.compute_loss(X_train, y_train)
            val_loss = self.model.compute_loss(X_val, y_val)

            train_accuracy = self.compute_accuracy(X_train, y_train)
            val_accuracy = self.compute_accuracy(X_val, y_val)

            # 存储指标
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)

            # 早停机制
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

                # 打印训练信息
            print(f"Epoch {epoch + 1}/{self.max_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}\n")

            # 早停
            if patience_counter >= self.early_stopping_patience:
                print("Early stopping triggered")
                break

        return self.train_losses, self.val_losses

    def plot_training_metrics(self):
        """
        可视化训练过程指标
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 5))

        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='训练集损失', color='blue')
        plt.plot(self.val_losses, label='验证集损失', color='red')
        plt.title('训练过程损失曲线')
        plt.xlabel('训练轮次')
        plt.ylabel('损失')
        plt.legend()

        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='训练集准确率', color='green')
        plt.plot(self.val_accuracies, label='验证集准确率', color='orange')
        plt.title('训练过程准确率曲线')
        plt.xlabel('训练轮次')
        plt.ylabel('准确率')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.close()