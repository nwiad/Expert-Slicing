import gensim
import torch
from moe import SentimentClassificationMoE
from dataset import SentimentTextDataset
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.metrics import f1_score
import torch.nn as nn
import deepspeed
import os

EMBEDDING_DIM = 50 # 词向量长度，不可调
HIDDEN_DIM = 50 # 隐含层的维度，需与词向量长度相同，不可调
OUTPUT_DIM = 2 # 分类数，不可调

TRUNCATION = 50 # 截断长度，可调
EPOCH = 10 # 训练轮数，可调
LEARNING_RATE = 1e-3 # 学习率，可调
BATCH_SIZE = 64 # 批次大小，可调
EP_SIZE = int(os.getenv('EP_SIZE')) # 由脚本参数设置
EXPERTS_NUM = 4 # 专家数量，可调，需要保证能被 EP_SIZE 整除

IN_FEATURES = HIDDEN_DIM # 需与隐含层维度相同，不可调
HIDDEN_FEATURES = 4 * HIDDEN_DIM # 可调
OUTPUT_FEATURES = HIDDEN_DIM # 不可调

AVAILABLE_DEVICES = [0, 1, 2, 3] # 可用 GPU 列表，可调，需为 [0, 1, 2, 3] 的子集
    
class SlicedFFN(nn.Module): # 专家数为 2，切片数为 2 
    def __init__(self, in_features, hidden_dim, out_features, device_ids):
        super().__init__()
        self.device_ids = device_ids
        self.fc_1 = nn.Linear(in_features=in_features, out_features=hidden_dim)
        self.fc_2 = nn.Linear(in_features=hidden_dim, out_features=out_features)
        # 将权重矩阵按行切分
        self.fc_1_weights_0, self.fc_1_weights_1 = torch.chunk(
            self.fc_1.weight,
            chunks=2,
            dim=0
        )
        # 将偏置矩阵按行切分
        self.fc_1_bias_0, self.fc_1_bias_1 = torch.chunk(
            self.fc_1.bias,
            chunks=2,
            dim=0
        )
        # 将权重矩阵按列切分
        self.fc_2_weights_0, self.fc_2_weights_1 = torch.chunk(
            self.fc_2.weight,
            chunks=2,
            dim=1
        )
        # 第二层的 bias 不需要切分，直接分配到两个 GPU 上即可

        # 将切分后的权重移动到相应的 GPU
        self.fc_1_weights_0 = nn.Parameter(self.fc_1_weights_0).to(device_ids[0])
        self.fc_1_weights_1 = nn.Parameter(self.fc_1_weights_1).to(device_ids[1])

        self.fc_1_bias_0 = nn.Parameter(self.fc_1_bias_0).to(device_ids[0])
        self.fc_1_bias_1 = nn.Parameter(self.fc_1_bias_1).to(device_ids[1])

        self.fc_2_weights_0 = nn.Parameter(self.fc_2_weights_0).to(device_ids[0])
        self.fc_2_weights_1 = nn.Parameter(self.fc_2_weights_1).to(device_ids[1])

        self.fc_2_bias_0 = nn.Parameter(self.fc_2.bias).to(device_ids[0])
        self.fc_2_bias_1 = nn.Parameter(self.fc_2.bias).to(device_ids[1])

    def forward(self, x):
        # 在两个 GPU 上分别执行对应的计算
        with torch.cuda.device(self.device_ids[0]):
            out_0 = F.linear(x.to(self.device_ids[0]), self.fc_1_weights_0.half(), self.fc_1_bias_0.half())
            out_0 = F.linear(F.relu(out_0), self.fc_2_weights_0.half(), self.fc_2_bias_0.half())

        with torch.cuda.device(self.device_ids[1]):
            out_1 = F.linear(x.to(self.device_ids[1]), self.fc_1_weights_1.half(), self.fc_1_bias_1.half())
            out_1 = F.linear(F.relu(out_1), self.fc_2_weights_1.half(), self.fc_2_bias_1.half())

        # 将输出相加
        out = out_0.to(x.device) + out_1.to(x.device)
        return out

def slice_expert():
    return SlicedFFN(IN_FEATURES, HIDDEN_FEATURES, OUTPUT_FEATURES, 4, AVAILABLE_DEVICES)

vec_path = "dataset/wiki_word2vec_50.bin"
train_path = "dataset/train.txt"
validation_path = "dataset/validation.txt"
save_path = "models/sliced_moe.pt"

# 读取词向量模型
vec = gensim.models.KeyedVectors.load_word2vec_format(vec_path, binary=True)

# 构建词汇表
key2index = vec.key_to_index
vectors = vec.vectors

def build_dataloader(dataset_path):
    labels = []
    texts = []
    max_length = -1

    # 读取训练集
    with open(dataset_path, "r", encoding="utf-8") as fin:
        while line := fin.readline():
            line = line.split("\t")
            # 读取标签
            labels.append(int(line[0]))
            # 读取评论
            texts.append(line[1].split())
            # 求最大长度
            max_length = max(max_length, len(texts[-1]))
        fin.close()

    # 设置截断
    max_length = min(max_length, TRUNCATION)

    padding_index = key2index["把"] # 110
    embedding_sentences = []
    for line in texts:
        sentence = []
        cnt = 0
        for c in line:
            sentence.append(key2index[c] if c in key2index.keys() else np.random.randint(len(key2index)))
            cnt += 1
            if cnt >= TRUNCATION:
                break
        while cnt < max_length:
            sentence.append(padding_index)
            cnt += 1
        tensor_sentence = torch.tensor(sentence)
        embedding_sentences.append(tensor_sentence)
        assert len(sentence) == max_length, "Bad Size"

    # 构建数据集
    dataset = SentimentTextDataset(labels, embedding_sentences)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

dataloader = build_dataloader(train_path)
validation_dataloader = build_dataloader(validation_path)

# 创建模型
device = torch.device("cuda")
model = SentimentClassificationMoE(
        vocab_size=len(key2index), embedding=vectors, embedding_dim=EMBEDDING_DIM, 
        expert=None, expert_constructor=slice_expert, ep_size = EP_SIZE,
        experts_num=EXPERTS_NUM, hidden_size=HIDDEN_DIM, output_dim=OUTPUT_DIM
    ).to(device)
model_engine, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=Adam(model.parameters(), lr=LEARNING_RATE),
                                                        model_parameters=model.parameters(), config="ds_config.json")
def convert(cuda_list: list):
    return torch.as_tensor(cuda_list).cpu()
# 训练
for i in range(EPOCH):
    model_engine.train()
    accuracies = []
    for index, (labels, matrixes) in enumerate(dataloader):
        prediction = model_engine(matrixes.to(device))
        loss = F.nll_loss(prediction, labels.to(device))
        result = torch.max(prediction, dim=1)[1]
        accuracy = torch.eq(result.to(device), labels.to(device)).float().mean()
        accuracies.append(accuracy)
        model_engine.backward(loss)
        model_engine.step()
    average = np.array(convert(accuracies)).mean()
    print("Epoch {epoch}: 准确率为{average}".format(epoch=i+1, average=average))

    if i%3 == 0:
        model_engine.eval()
        y_true = []
        y_pred = []
        accuracies = []
        with torch.no_grad():
            for labels, matrixes in dataloader:
                prediction = model_engine(matrixes.to(device))
                result = torch.max(prediction, dim=1)[1]
                accuracies.append(torch.eq(result.to(device), labels.to(device)).float().mean())
                y_true.append(labels)
                y_pred.append(result)
            y_true = convert(torch.cat(y_true, dim=0))
            y_pred = convert(torch.cat(y_pred, dim=0))
            accuracy = np.array(convert(accuracies)).mean()
            score = f1_score(y_true=y_true, y_pred=y_pred)
            print("验证：")
            print("准确率: {}".format(accuracy))
            print("f1-score: {}".format(score))

# 保存模型
torch.save(model.state_dict(), save_path)
