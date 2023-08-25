# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import copy
import os

# import torch.nn as nn

# HIDDEN_DIM = 50 # 隐含层的维度
# IN_FEATURES = HIDDEN_DIM
# HIDDEN_FEATURES = 4 * HIDDEN_DIM
# OUTPUT_FEATURES = HIDDEN_DIM

# class FFN(nn.Module):
#     def __init__(self, in_features, hidden_dim, out_features):
#         super().__init__()
#         self.fc1 = nn.Linear(in_features, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, out_features)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.1)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x
    
# def slice_expert():
#     return FFN(IN_FEATURES, HIDDEN_FEATURES, OUTPUT_FEATURES)

class Experts(torch.nn.Module):

    def __init__(self, expert, num_local_experts=1, expert_group_name=None, expert_constructor=None):
        super(Experts, self).__init__()

        if expert_constructor is None and os.getenv('EXPERT_SLICING') != '1':
            self.deepspeed_experts = torch.nn.ModuleList([copy.deepcopy(expert) for i in range(num_local_experts)])
            # self.deepspeed_experts = torch.nn.ModuleList([FFN(IN_FEATURES, HIDDEN_FEATURES, OUTPUT_FEATURES) for i in range(num_local_experts)])
            # self.deepspeed_experts = torch.nn.ModuleList([slice_expert() for i in range(num_local_experts)])
            # 疑问：多卡训练中，expert 为 17 行定义的 FFN 时，将 41 行替换为 42 行，效果一致，替换为 43 行则准确率明显下降
        elif expert_constructor is not None and os.getenv('EXPERT_SLICING') == '1':
            self.deepspeed_experts = torch.nn.ModuleList([expert_constructor() for i in range(num_local_experts)])

        self.num_local_experts = num_local_experts

        # TODO: revisit allreduce for moe.gate...
        for expert in self.deepspeed_experts:
            # TODO: Create param groups to handle expert + data case (e.g. param.group = moe_group)
            for name, param in expert.named_parameters():
                param.allreduce = False
                param.group_name = expert_group_name

    def forward(self, inputs):
        chunks = inputs.chunk(self.num_local_experts, dim=1)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.deepspeed_experts):
            out = expert(chunk)
            if type(out) is tuple:
                out = out[0]  # Ignore the bias term for now
            expert_outputs += [out]

        expert_output = torch.cat(expert_outputs, dim=1)
        return expert_output
