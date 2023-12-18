import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import get_time


# 定义一维卷积神经网络模型
class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)  # 添加ReLU激活函数
        x = self.fc2(x)
        return x


# 加载数据并转换为PyTorch张量
data_train = np.loadtxt('./pen+based+recognition+of+handwritten+digits/pendigits.tra', delimiter=',')
data_test = np.loadtxt('./pen+based+recognition+of+handwritten+digits/pendigits.tes', delimiter=',')

X_train = data_train[:, :-1]
y_train = data_train[:, -1]
X_test = data_test[:, :-1]
y_test = data_test[:, -1]

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # 在通道维度上添加一维
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)  # 在通道维度上添加一维
y_test = torch.tensor(y_test, dtype=torch.long)

# 创建模型实例
model = CNN1D()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 设置训练参数
num_epochs = 30
batch_size = 200

# 记录每个epoch的loss
losses = []


def draw_fig():
    plt.figure(figsize=(10, 6), dpi=800)
    plt.rcParams['backend'] = 'Agg'
    plt.plot(losses, color='blue', marker='o', linestyle='--', )
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Loss', fontweight='bold')
    plt.title('CNN Training Loss Curve', fontweight='bold')
    plt.grid(True, linestyle='dashed')  # 添加网格线

    plt.savefig('figs/cnn_pytorch.png')
    plt.show()


@get_time
def train_and_predict():
    for epoch in tqdm(range(num_epochs)):
        model.train()  # 设置模型为训练模式
        total_loss = 0.0

        # 批量训练
        for i in range(0, len(X_train), batch_size):
            inputs = X_train[i:i + batch_size]
            labels = y_train[i:i + batch_size]

            optimizer.zero_grad()  # 梯度清零

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 计算平均损失
        avg_loss = total_loss / (len(X_train) // batch_size)

        # 记录训练信息
        losses.append(avg_loss)


def evaluate_model():
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == y_test).float().sum().item()
        accuracy = correct / len(y_test)

    print(f'测试集准确性: {accuracy * 100:.2f}%')


def cnn_pytorch():
    train_and_predict()
    evaluate_model()
    draw_fig()
