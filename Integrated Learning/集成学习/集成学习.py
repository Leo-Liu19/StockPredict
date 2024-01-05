import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
# 假设CSV文件的列包括两个输入变量（input1, input2）和一个输出变量（output）
# 请根据你的实际数据调整列名和文件路径
csv_file_path = 'Dataset/CNN-GRU with score.csv'
df = pd.read_csv(csv_file_path)

# 提取输入和输出
# 这里假设列名为 'input1', 'input2', 'output'
# 请根据你的实际数据调整列名，TRUE代表的是实际的值
X = df[['CNN', 'GRU']].values
y = df['TRUE'].values

# 转换为PyTorch张量
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # 注意要将y变为列向量


# 定义神经网络模型
class BinaryClassificationNN(nn.Module):
    def __init__(self):
        super(BinaryClassificationNN, self).__init__()
        self.fc = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


# 设置输入和输出层的大小
input_size = 2

# 创建模型实例
model = BinaryClassificationNN()

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 划分训练集和测试集
# 这里简单地将数据划分为80%的训练集和20%的测试集
split_ratio = 0.8
split_index = int(split_ratio * len(X))

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 训练模型
epochs = 200
losses = [] # 用于绘图
for epoch in range(epochs):
    # 前向传播
    outputs = model(X_train)

    # 计算损失
    loss = criterion(outputs, y_train)
    losses.append(loss.item())

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 30 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

# 在测试集上评估模型
with torch.no_grad():
    test_outputs = model(X_test)
    predicted_labels = (test_outputs >= 0.5).float()  # 二分类阈值设为0.5
    accuracy = (predicted_labels == y_test).float().mean().item()

print(f'Test Accuracy: {accuracy * 100:.2f}%')

# 绘图
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs- CNN-GRU with score')
# plt.title('Training Loss Over Epochs-GRU without emotion')
plt.legend()
plt.savefig('image/CNN-GRU with score-training_loss_plot.png')  # 保存图像为PNG文件
# plt.savefig('image/GRU-withoutScore-training_loss_plot.png')  # 保存图像为PNG文件
plt.show()


