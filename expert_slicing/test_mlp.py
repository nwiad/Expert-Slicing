import gensim
import torch
from para_mlp import SentimentClassificationParallelMLP
from dataset import SentimentTextDataset
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.metrics import f1_score
from parallel_mlp.initialize import initialize_model_parallel
import os

TRUNCATION = 50 # 截断长度
EMBEDDING_DIM = 50 # 词向量长度
HIDDEN_DIM = 50 # 隐含层的维度
OUTPUT_DIM = 2 # 分类数
EPOCH = 10 # 训练轮数
LEARNING_RATE = 1e-3 # 学习率
BATCH_SIZE = 64 # 批次大小

vec_path = "dataset/wiki_word2vec_50.bin"
train_path = "dataset/train.txt"
validation_path = "dataset/validation.txt"

torch.distributed.init_process_group(backend='nccl')
world_size = int(os.environ['WORLD_SIZE'])
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
TP_SIZE = int(os.getenv('TP_SIZE'))
assert type(TP_SIZE) == int, "TP_SIZE must be int"
assert TP_SIZE >= 1 and TP_SIZE <= torch.cuda.device_count(), "TP_SIZE must be in [1, device_count]"
initialize_model_parallel(TP_SIZE)

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
model = SentimentClassificationParallelMLP(
    vocab_size=len(key2index), embedding=vectors, embedding_dim=EMBEDDING_DIM, 
    hidden_size=HIDDEN_DIM, ffn_hidden_size=BATCH_SIZE * HIDDEN_DIM, output_dim=OUTPUT_DIM
).to(device)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# 训练
for i in range(EPOCH):
    model.train()
    accuracies = []
    for index, (labels, matrixes) in enumerate(dataloader):
        prediction = model(matrixes.to(device))
        loss = F.nll_loss(prediction, labels.to(device))
        result = torch.max(prediction, dim=1)[1]
        accuracy = torch.eq(result.to(device), labels.to(device)).float().mean()
        accuracies.append(accuracy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    average = torch.mean(torch.as_tensor(accuracies))
    print("Epoch {epoch}: 准确率为{average}".format(epoch=i+1, average=average))

    if i%3 == 0:
        model.eval()
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
            y_true = torch.cat(y_true, dim=0)
            y_pred = torch.cat(y_pred, dim=0)
            accuracy = torch.mean(torch.as_tensor(accuracies))
            score = f1_score(y_true=torch.Tensor.cpu(y_true), y_pred=torch.Tensor.cpu(y_pred))
            print("验证：")
            print("准确率: {}".format(accuracy))
            print("f1-score: {}".format(score))