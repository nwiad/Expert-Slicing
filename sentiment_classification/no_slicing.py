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

TRUNCATION = 50 # 截断长度
EMBEDDING_DIM = 50 # 词向量长度
HIDDEN_DIM = 50 # 隐含层的维度
OUTPUT_DIM = 2 # 分类数
EPOCH = 10 # 训练轮数
LEARNING_RATE = 1e-3 # 学习率
BATCH_SIZE = 64 # 批次大小
EXPERTS_NUM = 1 # 专家数量

IN_FEATURES = HIDDEN_DIM
HIDDEN_FEATURES = 4 * HIDDEN_DIM
OUTPUT_FEATURES = HIDDEN_DIM

class FFN(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

vec_path = "dataset/wiki_word2vec_50.bin"
train_path = "dataset/train.txt"
validation_path = "dataset/validation.txt"
save_path = "models/moe.pt"

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
        expert=FFN(IN_FEATURES, HIDDEN_FEATURES, OUTPUT_FEATURES), 
        experts_num=EXPERTS_NUM, hidden_size=HIDDEN_DIM, output_dim=OUTPUT_DIM
    ).to(device)
model_engine, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=Adam(model.parameters(), lr=LEARNING_RATE),
                                                        model_parameters=model.parameters(), config="no_slicing_config.json")
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
    # for index, (labels, matrixes) in enumerate(dataloader):
    #     prediction = model(matrixes.to(device))
    #     loss = F.nll_loss(prediction, labels)
    #     result = torch.max(prediction, dim=1)[1]
    #     accuracy = torch.eq(result, labels).float().mean()
    #     accuracies.append(accuracy)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    average = np.array(convert(accuracies)).mean()
    print("Epoch {epoch}: 准确率为{average}".format(epoch=i+1, average=average))

    if i%3 == 0:
        model_engine.eval()
        y_true = []
        y_pred = []
        accuracies = []
        with torch.no_grad():
            for labels, matrixes in dataloader:
                prediction = model(matrixes.to(device))
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
