import argparse
import torch
# 创建一个ArgumentParser对象，用于解析命令行参数
parser = argparse.ArgumentParser()
# 添加参数：数据集文件的默认位置
parser.add_argument('--corpusFile', default='Dataset.csv')

# TODO 常改动参数
parser.add_argument('--gpu', default=0, type=int) # gpu 卡号
parser.add_argument('--epochs', default=100, type=int) # 训练轮数
parser.add_argument('--layers', default=1, type=int) #RNN层数
parser.add_argument('--input_size', default=7, type=int) #输入特征的维度（个数）
parser.add_argument('--hidden_size', default=32, type=int) #隐藏层的维度
parser.add_argument('--lr', default=0.0001, type=float) #learning rate 学习率
parser.add_argument('--sequence_length', default=5, type=int) # sequence的长度，默认是用前五天的数据来预测下一天的收盘价
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--useGPU', default=False, type=bool) #是否使用GPU
parser.add_argument('--batch_first', default=True, type=bool) #是否将batch_size放在第一维
parser.add_argument('--dropout', default=0.1, type=float)  # 丢弃率为0.1
parser.add_argument('--save_file', default='rnn_stock.pkl') # 模型保存位置

# 解析定义的参数
args = parser.parse_args()
# 设置设备，如果CUDA可用且用户选择使用GPU，则使用GPU；否则使用CPU
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.useGPU else "cpu")

args.device = device        # 将设备信息添加到args中，方便后续调用
