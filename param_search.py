import numpy as np
from model import NeuralNetwork
from trainer import Trainer


def simplified_hyperparameter_search(X_train, y_train, X_val, y_val):
    def random_config():
        hidden_sizes_options = [[256, 128], [512, 256], [1024, 512]]
        return {
            # 直接随机选择一个预定义的隐藏层配置
            'hidden_sizes': hidden_sizes_options[np.random.randint(len(hidden_sizes_options))],
            'activation': np.random.choice(['relu', 'leaky_relu', 'tanh']),
            'learning_rate': 10 ** np.random.uniform(-4, -2),
            'reg_strength': 10 ** np.random.uniform(-4, -1)
        }

    num_searches = 20
    best_config = None
    best_val_loss = float('inf')

    for _ in range(num_searches):
        config = random_config()

        model = NeuralNetwork(
            input_size=X_train.shape[1],
            hidden_sizes=config['hidden_sizes'],
            output_size=y_train.shape[1],
            activation=config['activation']
        )

        trainer = Trainer(
            model,
            learning_rate=config['learning_rate'],
            max_epochs=50
        )

        train_losses, val_losses = trainer.train(X_train, y_train, X_val, y_val)
        val_loss = val_losses[-1]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_config = config

    return best_config