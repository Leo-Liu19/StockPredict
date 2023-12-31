import argparse
import torch
'''
    参数列表
'''
parser = argparse.ArgumentParser()

parser.add_argument('--corpusFile', default='data/Datasetnewwith.csv')
# parser.add_argument('--corpusFile', default='data/000001SH_index.csv')


'''
8个特征为：
close ：收盘价
open：开盘价
high：最高价
low：最低价
change：涨跌额  去掉这个
pct_chg：涨跌幅  去掉这个
vol：成交量
amount：成交额 
'''
# TODO 常改动参数
parser.add_argument('--gpu', default=0, type=int) # gpu 卡号
parser.add_argument('--epochs', default=100, type=int) # 训练轮数
parser.add_argument('--layers', default=2, type=int) # LSTM层数
parser.add_argument('--input_size', default=7, type=int) #输入特征的维度
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