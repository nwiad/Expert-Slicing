import deepspeed
import torch
import torch.nn as nn

class Expert(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, device_ids):
        super().__init__()
        self.device_ids = device_ids

        # 将权重矩阵按行切分
        self.fc_1_weights_0, self.fc_1_weights_1 = torch.chunk(
            nn.Linear(in_features=in_features, out_features=hidden_dim).weight,
            chunks=2,
            dim=0
        )

        self.fc_2_weights_0, self.fc_2_weights_1 = torch.chunk(
            nn.Linear(in_features=hidden_dim, out_features=out_features).weight,
            chunks=2,
            dim=1
        )

        # 将切分后的权重移动到相应的 GPU
        self.fc_1_weights_0 = nn.Parameter(self.fc_1_weights_0).to(device_ids[0])
        self.fc_1_weights_1 = nn.Parameter(self.fc_1_weights_1).to(device_ids[1])

        self.fc_2_weights_0 = nn.Parameter(self.fc_2_weights_0).to(device_ids[0])
        self.fc_2_weights_1 = nn.Parameter(self.fc_2_weights_1).to(device_ids[1])

    def forward(self, x):

        # 在第一个 GPU 上执行第一部分的矩阵乘法
        with torch.cuda.device(self.device_ids[0]):
            out_0 = torch.matmul(x, self.fc_1_weights_0.t())
            out_0 = torch.matmul(nn.ReLU(out_0), self.fc_2_weights_0.t())

        # 在第二个 GPU 上执行第二部分的矩阵乘法
        with torch.cuda.device(self.device_ids[1]):
            out_1 = torch.matmul(x, self.fc_1_weights_1.t())
            out_1 = torch.matmul(nn.ReLU(out_1), self.fc_2_weights_1.t())

        # 将输出相加
        out = out_0 + out_1

        return out

# 利用deepspeed的moe模块，实现一个moe模型
class MoE(torch.nn.Module):
    def __init__(self, expert, hidden_size, experts_num, device_ids):
        super().__init__()
        self.experts_num = experts_num
        self.device_ids = device_ids
        self.moe_layer = deepspeed.moe.layer.MoE(hidden_size=hidden_size, expert=expert, num_experts=experts_num)

    def forward(self, x):
        return self.moe_layer(x)
    
device_ids = [0, 1]

h = 1024

IN = h
HIDDEN = 4 * h
OUT = h

# model = MoE(Expert(IN, HIDDEN, OUT, device_ids), IN, len(device_ids), device_ids)
    
model = MoE(Expert(IN, HIDDEN, OUT, device_ids), IN, len(device_ids), device_ids)
