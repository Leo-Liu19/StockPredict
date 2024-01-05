import torch.nn as nn

class rnn(nn.Module):
    # 构造函数初始化模型的各个层和参数
    def __init__(self, input_size=7, hidden_size=32, output_size=1, num_layers=1, dropout=0, batch_first=True):
        super(rnn, self).__init__()
        self.hidden_size = hidden_size  # 隐藏层大小
        self.input_size = input_size  # 输入层大小
        self.num_layers = num_layers  # RNN层的数量
        self.output_size = output_size  # 输出层大小
        self.dropout = dropout
        self.batch_first = batch_first
        # 定义RNN层，这里使用的是非门控RNN
        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout)
        # 定义全连接层，用于从RNN输出到最终预测结果的映射
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    # 前向传播函数定义了模型的前向传播逻辑
    def forward(self, x):
        out, _ = self.rnn(x)
        # 只取序列的最后一个时间点的输出用于预测
        out = self.fc(out[:, -1, :])        # out的形状为(batch_size, output_size)
        return out
