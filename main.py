import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from model import NeuralNetwork
from trainer import Trainer
from test import test_model
from param_search import simplified_hyperparameter_search


def load_cifar10(data_dir, sample_ratio=0.1):
    def unpickle(file):
        with open(file, 'rb') as fo:
            return pickle.load(fo, encoding='bytes')

    X_train, y_train = [], []
    for i in range(1, 6):
        file_path = os.path.join(data_dir, f'data_batch_{i}')
        batch_data = unpickle(file_path)

        # 只加载部分数据
        sample_indices = np.random.choice(
            len(batch_data[b'data']),
            int(len(batch_data[b'data']) * sample_ratio),
            replace=False
        )
        X_train.append(batch_data[b'data'][sample_indices])
        y_train.extend([batch_data[b'labels'][idx] for idx in sample_indices])

        # 加载测试数据
    test_file = os.path.join(data_dir, 'test_batch')
    test_data = unpickle(test_file)
    X_test = test_data[b'data']
    y_test = test_data[b'labels']

    # 重塑和归一化
    X_train = np.vstack(X_train).reshape(-1, 3072) / 255.0
    X_test = X_test.reshape(-1, 3072) / 255.0

    # One-hot编码
    y_train_onehot = np.eye(10)[y_train]
    y_test_onehot = np.eye(10)[y_test]

    return X_train, y_train_onehot, X_test, y_test_onehot, y_test


def main():
    # 设置随机种子以保证可重复性
    np.random.seed(42)

    # 加载数据
    data_dir = './cifar-10-batches-py'
    X_train, y_train, X_test, y_test_onehot, y_test = load_cifar10(data_dir, sample_ratio=0.2)

    # 划分验证集
    val_size = int(0.2 * len(X_train))
    X_val = X_train[:val_size]
    y_val = y_train[:val_size]
    X_train = X_train[val_size:]
    y_train = y_train[val_size:]

    # 超参数搜索
    best_config = simplified_hyperparameter_search(X_train, y_train, X_val, y_val)
    print("最佳配置:", best_config)

    # 使用最佳配置创建模型
    model = NeuralNetwork(
        input_size=X_train.shape[1],
        hidden_sizes=best_config['hidden_sizes'],
        output_size=y_train.shape[1],
        activation=best_config['activation']
    )

    # 训练
    trainer = Trainer(
        model,
        learning_rate=best_config['learning_rate'],
        max_epochs=20  # 减少训练轮次
    )
    train_losses, val_losses = trainer.train(X_train, y_train, X_val, y_val)

    # 测试
    test_metrics = test_model(model, X_test, y_test_onehot)

    # 打印结果
    print("\n--- 测试指标 ---")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value}")

        # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title('训练过程损失曲线')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.tight_layout()
    plt.savefig('loss_curve.png')
    plt.close()

    # 保存模型参数
    import json
    with open('best_model_config.json', 'w') as f:
        json.dump(best_config, f)

    print("\n模型训练完成。最佳配置已保存到 'best_model_config.json'")
    print("损失曲线已保存到 'loss_curve.png'")


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    end_time = time.time()
    print(f"总运行时间: {(end_time - start_time) / 60:.2f}分钟")