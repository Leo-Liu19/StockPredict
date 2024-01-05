from pandas import read_csv
import numpy as np
from torch.utils.data import DataLoader,Dataset
import torch
from torchvision import transforms
import torch.nn as nn
import argparse
from torch.autograd import Variable


#
def getData(corpusFile,sequence_length,batchSize):  # batchSize设置为64
    # 数据预处理 ，去除id、股票代码、前一天的收盘价、交易日期等对训练无用的无效数据
    stock_data = read_csv(corpusFile)
    stock_data.drop('ts_code', axis=1, inplace=True)  # 删除第二列’股票代码‘
    stock_data.drop('id', axis=1, inplace=True)  # 删除第一列’id‘
    stock_data.drop('pre_close', axis=1, inplace=True)  # 删除列’pre_close‘
    stock_data.drop('trade_date', axis=1, inplace=True)  # 删除列’trade_date‘
    #stock_data.drop('change', axis=1, inplace=True)
    #stock_data.drop('pct_chg', axis=1, inplace=True)


    close_max = stock_data['close'].max() #收盘价的最大值
    close_min = stock_data['close'].min() #收盘价的最小值
    df = stock_data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))  # min-max标准化

    # 构造X和Y
    #根据前n天的数据，预测未来一天的收盘价(close)， 例如：根据1月1日、1月2日、1月3日、1月4日、1月5日的数据（每一天的数据包含8个特征），预测1月6日的收盘价。
    sequence = sequence_length # 来自于参数 默认是用前五天的数据来预测下一天的收盘价
    X = []
    Y = []
    for i in range(df.shape[0] - sequence): # 按组遍历所有数据
        X.append(np.array(df.iloc[i:(i + sequence), ].values, dtype=np.float32)) # 每sequence为一组加到列表中
        Y.append(np.array(df.iloc[(i + sequence), 0], dtype=np.float32)) # 第六天的收盘价

    # 构建batch
    total_len = len(Y)
    # print(total_len)

    trainx, trainy = X[:int(0.70 * total_len)], Y[:int(0.70 * total_len)] # 99%的数据作为训练集
    testx, testy = X[int(0.30 * total_len):], Y[int(0.30 * total_len):] # 1%的数据作为测试集
    train_loader = DataLoader(dataset=Mydataset(trainx, trainy, transform=transforms.ToTensor()), batch_size=batchSize,
                              shuffle=True) # 创建数据加载器
    test_loader = DataLoader(dataset=Mydataset(testx, testy), batch_size=batchSize, shuffle=True) # 为什么这个不用transform？
    return close_max,close_min,train_loader,test_loader



class Mydataset(Dataset):
    def __init__(self, xx, yy, transform=None):
        self.x = xx
        self.y = yy
        self.tranform = transform

    def __getitem__(self, index):
        x1 = self.x[index]
        y1 = self.y[index]
        if self.tranform != None:
            return self.tranform(x1), y1
        return x1, y1

    def __len__(self):
        return len(self.x)
class GRUModel(nn.Module):

    def __init__(self, input_size=8, hidden_size=32, num_layers=1 , output_size=1 , dropout=0, batch_first=True):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.batch_first = batch_first

        # 使用 GRU 替换 LSTM
        self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        # GRU 只返回输出和隐藏状态
        out, _ = self.rnn(x)
        out = self.linear(out[:, -1, :])  # 只取序列的最后一个输出
        return out


parser = argparse.ArgumentParser()

parser.add_argument('--corpusFile', default='data/000001SH_index.csv')


# TODO 常改动参数
parser.add_argument('--gpu', default=0, type=int) # gpu 卡号
parser.add_argument('--epochs', default=100, type=int) # 训练轮数
parser.add_argument('--layers', default=2, type=int) # LSTM层数
parser.add_argument('--input_size', default=8, type=int) #输入特征的维度
parser.add_argument('--hidden_size', default=32, type=int) #隐藏层的维度
parser.add_argument('--lr', default=0.0001, type=float) #learning rate 学习率
parser.add_argument('--sequence_length', default=5, type=int) # sequence的长度，默认是用前五天的数据来预测下一天的收盘价
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--useGPU', default=False, type=bool) #是否使用GPU
parser.add_argument('--batch_first', default=True, type=bool) #是否将batch_size放在第一维
parser.add_argument('--dropout', default=0.1, type=float)  # 丢弃率为0.1
parser.add_argument('--save_file', default='model/stock.pkl') # 模型保存位置


args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.useGPU else "cpu")
args.device = device
def train():

    # 使用 GRUModel 替换原来的 lstm
    model = GRUModel(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers, output_size=1, dropout=args.dropout, batch_first=args.batch_first)
    model.to(args.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    close_max, close_min, train_loader, test_loader = getData(args.corpusFile,args.sequence_length,args.batch_size ) # 详情请进入方法
    for i in range(args.epochs):
        total_loss = 0
        for idx, (data, label) in enumerate(train_loader): # 遍历每个批次
            if args.useGPU: # 可以先放放，我们不用cuda
                data1 = data.squeeze(1).cuda()
                pred = model(Variable(data1).cuda())
                # print(pred.shape)
                pred = pred[1,:,:]
                label = label.unsqueeze(1).cuda()
                # print(label.shape)
            else:
                data1 = data.squeeze(1)
                pred = model(Variable(data1)) # 输入进模型获得预测
                pred = pred[1, :, :]
                label = label.unsqueeze(1)
            loss = criterion(pred, label)   # 计算损失函数
            optimizer.zero_grad()      # 梯度清零
            loss.backward()     #  反向传播
            optimizer.step() # 更新模型参数
            total_loss += loss.item() # 计算整个训练过程的总损失
        print(total_loss)  # 打印本轮训练的总损失
        if i % 10 == 0:  # 每十轮保存一次模型
            # torch.save(model, args.save_file)
            torch.save({'state_dict': model.state_dict()}, args.save_file) # 将模型的状态字典保存到文件中
            print('第%d epoch，保存模型' % i)
    # torch.save(model, args.save_file)
    torch.save({'state_dict': model.state_dict()}, args.save_file)  # 保存最终模型

train()