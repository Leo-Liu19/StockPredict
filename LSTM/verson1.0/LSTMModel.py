import torch.nn as nn


class lstm(nn.Module): # 将 nn.Module 放在括号里表示 lstm 类继承自 nn.Module

    def __init__(self, input_size=8, hidden_size=32, num_layers=1 , output_size=1 , dropout=0, batch_first=True):
        super(lstm, self).__init__() # lstm 和 self都是在寻找父类
        # lstm的输入 #batch,seq_len, input_size
        self.hidden_size = hidden_size # 隐藏层大小
        self.input_size = input_size # 输入层大小
        self.num_layers = num_layers # LSTM层的数量
        self.output_size = output_size # 输出层大小
        self.dropout = dropout
        self.batch_first = batch_first
        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout )
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        out, (hidden, cell) = self.rnn(x)  # x.shape : batch, seq_len, hidden_size , hn.shape and cn.shape : num_layes * direction_numbers, batch, hidden_size
        # a, b, c = hidden.shape
        # out = self.linear(hidden.reshape(a * b, c))
        out = self.linear(hidden)
        return out