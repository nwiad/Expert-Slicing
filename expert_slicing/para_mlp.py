from layers import ParallelMLP
import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentClassificationParallelMLP(nn.Module):
    def __init__(self, vocab_size, embedding, embedding_dim, hidden_size, ffn_hidden_size, output_dim):
        super().__init__()
        # 设置全局嵌入层以供索引
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # 嵌入层加入训练
        self.embedding.weight.requires_grad = True
        # 利用词向量模型进行初始化
        tensor_embedding = torch.stack([torch.from_numpy(array).float() for array in embedding])
        self.embedding.weight.data.copy_(tensor_embedding)
        self.para_mlp = ParallelMLP(hidden_size, ffn_hidden_size)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        embedded = self.embedding(x).mean(dim=1)
        output, output_bias = self.para_mlp(embedded)
        return F.log_softmax(self.fc(output+output_bias), dim=1)
