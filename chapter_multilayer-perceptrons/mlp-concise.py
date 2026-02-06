# 多层感知机的简洁实现
import sys
import os

# 添加虚拟环境路径到sys.path，以便能够找到d2l模块
venv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.venv', 'Lib', 'site-packages')
if os.path.exists(venv_path):
    sys.path.insert(0, venv_path)

import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

## 模型
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

## [**训练过程**]的实现与我们实现softmax回归时完全相同，
# 这种模块化设计使我们能够将与模型架构有关的内容独立出来。

batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)

if __name__ == '__main__':
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # 自定义训练循环，添加进度打印
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                            legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        train_metrics = d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        test_acc = d2l.evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))

        # 打印训练进度
        train_loss, train_acc = train_metrics
        print(f'epoch {epoch + 1}, loss {train_loss:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')

    # 最终验证
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

    # 显示训练图表
    print('\n训练完成！显示训练曲线...')
    plt.show()
