import torch.nn as nn

# 定义 GRUModel 类，继承自 PyTorch 的 nn.Module 类
class GRUModel(nn.Module):
    # 初始化方法
    def __init__(self, input_size=8, hidden_size=32, num_layers=1, output_size=1, dropout=0, batch_first=True):
        super(GRUModel, self).__init__()  # 调用父类的初始化方法
        # 初始化传入的参数
        self.hidden_size = hidden_size  # 隐藏层的大小
        self.input_size = input_size    # 输入层的大小
        self.num_layers = num_layers    # GRU层的数量
        self.output_size = output_size  # 输出层的大小
        self.dropout = dropout          # Dropout比率
        self.batch_first = batch_first  # 数据的批次是否在前

        # 定义GRU层
        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=self.batch_first,
                          dropout=self.dropout)
        # 定义线性层，用于从GRU的输出到最终的输出
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    # 定义前向传播方法
    def forward(self, x):
        # 通过GRU层进行计算，它返回每个时间步的输出和最后一个时间步的隐藏状态
        out, _ = self.rnn(x)
        # 只取序列的最后一个时间步的输出，用于进行最终的线性变换
        out = self.linear(out[:, -1, :])
        return out