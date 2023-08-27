import numpy as np
import os

sliced = []
unsliced = []
with open("sliced.txt", 'r') as f:
    while line:=f.readline():
        sliced.append(float(line))
sliced_time = np.mean(sliced)
with open("unsliced.txt", 'r') as f:
    while line:=f.readline():
        unsliced.append(float(line))
unsliced_time = np.mean(unsliced)
print("sliced:", sliced_time)
print("unsliced:", unsliced_time)
print("ratio:", sliced_time / unsliced_time)
res = {}
res['HIDEN_DIM'] = int(os.getenv('HIDDEN_DIM'))
res['sliced_time'] = sliced_time
res['unsliced_time'] = unsliced_time
res['ratio'] = sliced_time / unsliced_time
with open("results.txt","a") as f:
    f.write(str(res) + "\n")