# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import copy
import torch.nn as nn
import os

in_features = 1024
hidden_dim = 1024
out_features = 2

class Experts(torch.nn.Module):

    def __init__(self, expert, num_local_experts=1, expert_group_name=None):
        super(Experts, self).__init__()

        self.expert_slicing = os.getenv('EXPERT_SLICING') == '1'

        if self.expert_slicing:
            assert num_local_experts == 1, "num_local_experts must be 1 for expert slicing"
            self.device_ids = [0,1]

            # 将权重矩阵按行切分
            self.fc_1_weights_0, self.fc_1_weights_1 = torch.chunk(
                nn.Linear(in_features=in_features, out_features=hidden_dim).weight,
                chunks=2,
                dim=0
            )

            # 将切分后的权重移动到相应的 GPU
            self.fc_1_weights_0 = nn.Parameter(self.fc_1_weights_0).to(self.device_ids[0])
            self.fc_1_weights_1 = nn.Parameter(self.fc_1_weights_1).to(self.device_ids[1])

            self.fc_2_weights_0, self.fc_2_weights_1 = torch.chunk(
                nn.Linear(in_features=hidden_dim, out_features=out_features).weight,
                chunks=2,
                dim=1
            )

            self.fc_2_weights_0 = nn.Parameter(self.fc_2_weights_0).to(self.device_ids[0])
            self.fc_2_weights_1 = nn.Parameter(self.fc_2_weights_1).to(self.device_ids[1])
        else:
            self.deepspeed_experts = torch.nn.ModuleList([copy.deepcopy(expert) for i in range(num_local_experts)])
            self.num_local_experts = num_local_experts

            # TODO: revisit allreduce for moe.gate...
            for expert in self.deepspeed_experts:
                # TODO: Create param groups to handle expert + data case (e.g. param.group = moe_group)
                for name, param in expert.named_parameters():
                    param.allreduce = False
                    param.group_name = expert_group_name

    def forward(self, inputs):
        if self.expert_slicing:
            # 在第一个 GPU 上执行第一部分的矩阵乘法
            with torch.cuda.device(self.device_ids[0]):
                out_0 = torch.matmul(input, self.fc_1_weights_0.t())
                out_0 = torch.matmul(nn.ReLU(out_0), self.fc_2_weights_0.t())

            # 在第二个 GPU 上执行第二部分的矩阵乘法
            with torch.cuda.device(self.device_ids[1]):
                out_1 = torch.matmul(input, self.fc_1_weights_1.t())
                out_1 = torch.matmul(nn.ReLU(out_1), self.fc_2_weights_1.t())

            # 将输出相加
            out = out_0 + out_1

            return out
        else:
            chunks = inputs.chunk(self.num_local_experts, dim=1)
            expert_outputs = []
            for chunk, expert in zip(chunks, self.deepspeed_experts):
                out = expert(chunk)
                if type(out) is tuple:
                    out = out[0]  # Ignore the bias term for now
                expert_outputs += [out]

            expert_output = torch.cat(expert_outputs, dim=1)
            return expert_output
