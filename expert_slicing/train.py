import torch
from moe import MoE
from dataset import FakeDataSet
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from initialize import initialize_model_parallel
import os
from layers import ParallelMLP
import deepspeed

BATCH_SIZE = 64
HIDDEN_DIM = 1024
EXPERT_NUM = 4
EP_SIZE = int(os.getenv('EP_SIZE'))
LENGTH = 10000
LEARNING_RATE = 1e-3
EPOCH = 10

if os.getenv('EXPERT_SLICING') == '1':
    save_path = "models/sliced.pt"
elif os.getenv('EXPERT_SLICING') == '0':
    save_path = "models/unsliced.pt"

torch.distributed.init_process_group(backend='nccl')
world_size = int(os.environ['WORLD_SIZE'])
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
TP_SIZE = int(os.getenv('TP_SIZE'))
assert type(TP_SIZE) == int, "TP_SIZE must be int"
assert TP_SIZE >= 1 and TP_SIZE <= torch.cuda.device_count(), "TP_SIZE must be in [1, device_count]"
initialize_model_parallel(TP_SIZE)

def slice_expert():
    return ParallelMLP(hidden_size=HIDDEN_DIM, ffn_hidden_size=BATCH_SIZE * HIDDEN_DIM)

dataset = FakeDataSet(LENGTH, HIDDEN_DIM)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
device = torch.device('cuda')
model = MoE(
    expert=None,
    hidden_size=HIDDEN_DIM,
    ep_size=EP_SIZE,
    experts_num=EXPERT_NUM,
    expert_constructor=slice_expert
).to(device)
model, optimizer, _, _ = deepspeed.initialize(
    model=model, optimizer=Adam(model.parameters(), lr=LEARNING_RATE),
    model_parameters=model.parameters(), config="ds_config.json"
)

# for i in range(EPOCH):
#     model.train()
#     for index, (labels, matrixes) in enumerate(dataloader):
#         prediction = model(matrixes.to(device))
#         # print(prediction.shape, labels.shape)
#         loss = torch.sum(prediction) / 1000
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

torch.save(model.state_dict(), save_path)