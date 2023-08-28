import torch
from moe import MoE
from dataset import FakeDataSet
from torch.utils.data import DataLoader
from initialize import initialize_model_parallel
import os
from layers import ParallelMLP
import deepspeed
import time

BATCH_SIZE = 1
HIDDEN_DIM = int(os.getenv('HIDDEN_DIM', 5000))
print("Setting HIDDEN_DIM to", HIDDEN_DIM)
EXPERT_NUM = int(os.getenv('NUM_EXPERT'))
EP_SIZE = int(os.getenv('EP_SIZE'))
LENGTH = 10000
LEARNING_RATE = 1e-3

torch.distributed.init_process_group(backend='nccl')
world_size = int(os.environ['WORLD_SIZE'])
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
TP_SIZE = int(os.getenv('TP_SIZE'))
assert type(TP_SIZE) == int, "TP_SIZE must be int"
assert TP_SIZE >= 1 and TP_SIZE <= torch.cuda.device_count(), "TP_SIZE must be in [1, device_count]"
print(f"W{EP_SIZE} E{EXPERT_NUM} TP{TP_SIZE}")
initialize_model_parallel(TP_SIZE)

def slice_expert():
    return ParallelMLP(hidden_size=HIDDEN_DIM, ffn_hidden_size=4 * HIDDEN_DIM)

dataset = FakeDataSet(LENGTH, HIDDEN_DIM)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print("len(dataloader)", len(dataloader))
device = torch.device('cuda')
model = MoE(
    expert=None,
    hidden_size=HIDDEN_DIM,
    ep_size=EP_SIZE,
    experts_num=EXPERT_NUM,
    expert_constructor=slice_expert
).to(device)
model, _, _, _ = deepspeed.initialize(model=model, config="infer_config.json")
model.eval()
inference_time = 0
with torch.no_grad():
    for labels, matrixes in dataloader:
        start = time.time()
        prediction = model(matrixes.to(device))
        end = time.time()
        inference_time += end - start
if os.getenv('EXPERT_SLICING') == '1':
    filename = "sliced.txt"
elif os.getenv('EXPERT_SLICING') == '0':
    filename = "unsliced.txt"
with open(filename, 'a') as f:
    f.write(f"{inference_time / len(dataloader)}\n")
print(f"Average Inference Time: {inference_time / len(dataloader)}")
