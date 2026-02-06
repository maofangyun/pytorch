# 多层感知机的从零开始实现

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

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

## 初始化模型参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1, b1, W2, b2]

## 激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

## 模型
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
    return (H@W2 + b2)

## 损失函数
loss = nn.CrossEntropyLoss()

## 训练
if __name__ == '__main__':
    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(params, lr=lr)
    # 自定义训练循环，添加进度打印
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                            legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        train_metrics = d2l.train_epoch_ch3(net, train_iter, loss, updater)
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

