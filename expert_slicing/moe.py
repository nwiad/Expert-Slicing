import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentClassificationMoE(nn.Module):
    def __init__(self, vocab_size, embedding, embedding_dim, expert, hidden_size, ep_size, experts_num, output_dim, expert_constructor=None):
        """
        vocab_szie: 词向量总数/词表长度
        embedding: 训练好的词向量
        embedding_dim: 词向量长度，也是每个卷积核的宽度
        expert: 专家模型
        hidden_size: 输入维和隐藏维
        ep_size: 专家并行度
        experts_num: 专家数量
        output_dim: 分类数目
        expert_constructor: 专家构造器
        """
        super().__init__()
        # 设置全局嵌入层以供索引
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # 嵌入层加入训练
        self.embedding.weight.requires_grad = True
        # 利用词向量模型进行初始化
        tensor_embedding = torch.stack([torch.from_numpy(array).float() for array in embedding])
        self.embedding.weight.data.copy_(tensor_embedding)
        self.experts_num = experts_num
        if expert is not None and not callable(expert_constructor):
            self.moe_layer = deepspeed.moe.layer.MoE(hidden_size=hidden_size, expert=expert, ep_size=experts_num, num_experts=experts_num, min_capacity=0)
        elif callable(expert_constructor):
            assert expert is None, "expert and expert_constructor can't be set at the same time"
            self.moe_layer = deepspeed.moe.layer.MoE(hidden_size=hidden_size, expert=None, expert_constructor=expert_constructor, ep_size=ep_size, num_experts=experts_num, min_capacity=0)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        embedded = self.embedding(x).mean(dim=1)
        moe_output, _, _ = self.moe_layer(embedded)
        return F.log_softmax(self.fc(moe_output), dim=1)
    
class MoE(nn.Module):
    def __init__(self, expert, hidden_size, ep_size, experts_num, expert_constructor=None):
        """
        expert: 专家模型
        hidden_size: 输入维和隐藏维
        ep_size: 专家并行度
        experts_num: 专家数量
        expert_constructor: 专家构造器
        """
        super().__init__()
        self.experts_num = experts_num
        if expert is not None and not callable(expert_constructor):
            self.moe_layer = deepspeed.moe.layer.MoE(hidden_size=hidden_size, expert=expert, ep_size=experts_num, num_experts=experts_num, min_capacity=0)
        elif callable(expert_constructor):
            assert expert is None, "expert and expert_constructor can't be set at the same time"
            self.moe_layer = deepspeed.moe.layer.MoE(hidden_size=hidden_size, expert=None, expert_constructor=expert_constructor, ep_size=ep_size, num_experts=experts_num, min_capacity=0)
    
    def forward(self, x):
        """
        x.shape[-1] = hidden_size, output.shape[-1] = hidden_size
        """
        # moe_output, cal_time= self.moe_layer(x)
        moe_output, _, _ = self.moe_layer(x)
        # return moe_output, cal_time
        return moe_output