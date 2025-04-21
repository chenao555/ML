import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def test_model(model, X_test, y_test):
    # 预测
    y_pred = model.predict(X_test)

    # 检查 y_pred 的形状
    print("y_pred 形状:", y_pred.shape)
    print("y_test 形状:", y_test.shape)

    # 确保 y_pred 是二维的
    if y_pred.ndim == 1:
        # 如果是一维,尝试转换
        y_pred = np.eye(10)[y_pred]

        # 准确率
    try:
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))
    except Exception as e:
        print(f"准确率计算错误: {e}")
        # 降级处理
        accuracy = np.mean(y_pred == np.argmax(y_test, axis=1))

        # 混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

    # 可视化和分析(可选)
    try:
        visualize_network_parameters(model)
        analyze_network_parameters(model)
    except Exception as e:
        print(f"可视化分析出错: {e}")

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm
    }


# 其他可视化函数保持不变
def visualize_network_parameters(model):
    """
    可视化神经网络参数
    """
    plt.figure(figsize=(20, 15))

    # 1. 权重分布热图
    plt.subplot(2, 2, 1)
    layer_weights = []

    # 遍历所有权重矩阵
    for i in range(1, len(model.hidden_sizes) + 2):
        weights = model.params[f'W{i}']
        # 确保将权重展平
        layer_weights.append(weights.flatten())

        # 使用列表推导式确保所有权重都被展平
    layer_weights_array = [w.flatten() for w in [model.params[f'W{i}'] for i in range(1, len(model.hidden_sizes) + 2)]]

    # 使用pad_sequences或手动填充
    max_len = max(len(w) for w in layer_weights_array)
    layer_weights_padded = np.array([
        np.pad(w, (0, max_len - len(w)), mode='constant')
        for w in layer_weights_array
    ])

    sns.heatmap(
        layer_weights_padded,
        cmap='viridis',
        center=0,
        annot=False,
        cbar_kws={'label': '权重值'}
    )
    plt.title('网络层权重热图')

    # 2. 权重统计直方图
    plt.subplot(2, 2, 2)
    all_weights = np.concatenate(
        [w.flatten() for w in [model.params[f'W{i}'] for i in range(1, len(model.hidden_sizes) + 2)]])
    plt.hist(all_weights, bins=50, color='skyblue', edgecolor='black')
    plt.title('权重全局分布直方图')
    plt.xlabel('权重值')
    plt.ylabel('频率')

    # 3. 层间权重关联性
    plt.subplot(2, 2, 3)
    # 使用padded数组计算相关性
    weight_correlations = np.corrcoef(layer_weights_padded)
    sns.heatmap(
        weight_correlations,
        annot=True,
        cmap='coolwarm',
        center=0,
        cbar_kws={'label': '权重相关性'}
    )
    plt.title('网络层权重相关性')

    # 4. 权重范数变化
    plt.subplot(2, 2, 4)
    weight_norms = [np.linalg.norm(model.params[f'W{i}']) for i in range(1, len(model.hidden_sizes) + 2)]
    plt.bar(range(len(weight_norms)), weight_norms)
    plt.title('各层权重范数')
    plt.xlabel('网络层')
    plt.ylabel('权重范数')

    plt.tight_layout()
    plt.savefig('network_parameters_visualization.png')
    plt.close()


def analyze_network_parameters(model):
    """
    深入分析网络参数
    """
    print("网络参数分析报告:")

    for i in range(1, len(model.hidden_sizes) + 2):
        weights = model.params[f'W{i}']

        print(f"\n第 {i} 层权重分析:")
        print(f"  - 权重形状: {weights.shape}")
        print(f"  - 均值: {np.mean(weights):.4f}")
        print(f"  - 标准差: {np.std(weights):.4f}")
        print(f"  - 最小值: {np.min(weights):.4f}")
        print(f"  - 最大值: {np.max(weights):.4f}")

        # 检测异常值（超过3个标准差）
        mean = np.mean(weights)
        std = np.std(weights)
        outliers = np.sum((weights < mean - 3 * std) | (weights > mean + 3 * std))
        print(f"  - 异常值数量: {outliers}")