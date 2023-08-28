import matplotlib.pyplot as plt
with open("res/results.txt", "r") as f:
    results = []
    while line:=f.readline():
        try:
            results.append(eval(line))
        except:
            continue

x = []
y = []
sliced_time = []
unsliced_time = []
for res in results:
    x.append(res['HIDDEN_DIM'])
    y.append(res['ratio'])
    sliced_time.append(res['sliced_time'])
    unsliced_time.append(res['unsliced_time'])
x, y = zip(*sorted(zip(x, y)))
x, sliced_time = zip(*sorted(zip(x, sliced_time)))
x, unsliced_time = zip(*sorted(zip(x, unsliced_time)))
# 散点图
plt.scatter(x, y)
plt.plot([0, 16500], [0.625, 0.625], color='red', linewidth=2.0, linestyle='--')
plt.text(16500, 0.625, 'theoretical limit', ha='right', va='bottom', fontsize=10)
plt.xlabel("HIDDEN_DIM")
plt.ylabel("Ratio of Inference Time")
plt.savefig("pics/scatter.png")

plt.figure()
plt.ylim(0, 0.014)
plt.xlabel("HIDDEN_DIM")
plt.ylabel("Inference Time")
p1 = plt.scatter(x, sliced_time)
p2 = plt.scatter(x, unsliced_time)
plt.legend([p1, p2], ["Sliced MoE", "Unsliced MoE"])
plt.savefig("pics/sep_scatter.png")
# 折线图
plt.figure()
plt.plot(x, y)
plt.plot([0, 16500], [0.625, 0.625], color='red', linewidth=2.0, linestyle='--')
plt.text(16500, 0.625, 'theoretical limit', ha='right', va='bottom', fontsize=10)
plt.xlabel("HIDDEN_DIM")
plt.ylabel("Ratio of Inference Time")
plt.savefig("pics/line.png")

plt.figure()
plt.ylim(0, 0.014)
plt.xlabel("HIDDEN_DIM")
plt.ylabel("Inference Time")
p1, =plt.plot(x, sliced_time)
p2, =plt.plot(x, unsliced_time)
plt.legend([p1, p2], ["Sliced MoE", "Unsliced MoE"])
plt.savefig("pics/sep_line.png")

