from pandas import read_csv
import numpy as np
from torch.utils.data import DataLoader,Dataset
import torch
from torchvision import transforms
from parser_my import args

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
