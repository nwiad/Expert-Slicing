import torch
from moe import MoE
from dataset import FakeDataSet
from torch.utils.data import DataLoader
from parallel_mlp.initialize import initialize_model_parallel
import os
from parallel_mlp.layers import ParallelMLP
import deepspeed
import time

BATCH_SIZE = 8
HIDDEN_DIM = int(os.getenv('HIDDEN_DIM', 5000))
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

initialize_model_parallel(TP_SIZE)

def slice_expert():
    return ParallelMLP(hidden_size=HIDDEN_DIM, ffn_hidden_size=4 * HIDDEN_DIM)

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
model, _, _, _ = deepspeed.initialize(model=model, config="infer_config.json")
print(f"CUDA {local_rank}: HIDDEN_DIM={HIDDEN_DIM} W{EP_SIZE} E{EXPERT_NUM} TP{TP_SIZE}")
print(f"CUDA {local_rank}: len(dataloader)={len(dataloader)}")
model.eval()
inference_time = 0
cnt = 0
with torch.no_grad():
    print(f"CUDA {local_rank}: warming up")
    for labels, matrixes in dataloader:
        prediction = model(matrixes.to(device))
        cnt += 1
        if cnt % 25 == 0:
            print(f"CUDA {local_rank}: warm up: {cnt}%")
        if cnt == 100:
            break

    print(f"CUDA {local_rank}: warmed up")
    cnt = 0
    for labels, matrixes in dataloader:
        cnt += 1
        if cnt % 1000 == 0:
            print(f"CUDA {local_rank}: {cnt / len(dataloader) * 100}%")
        torch.cuda.synchronize()
        start = time.time()
        prediction = model(matrixes.to(device))
        torch.cuda.synchronize()
        end = time.time()
        inference_time += end - start
if os.getenv('EXPERT_SLICING') == '1':
    filename = "bin/sliced.txt"
elif os.getenv('EXPERT_SLICING') == '0':
    filename = "bin/unsliced.txt"
with open(filename, 'a') as f:
    f.write(f"{inference_time / len(dataloader)}\n")
print(f"CUDA {local_rank}: Average Inference Time={inference_time / len(dataloader)}s")
