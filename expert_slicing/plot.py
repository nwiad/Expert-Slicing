import matplotlib.pyplot as plt
with open("results.txt", "r") as f:
    res = []
    while line:=f.readline():
        try:
            res.append(eval(line))
        except:
            continue
x = []
y = []
for res_dict in res:
    x.append(res_dict['HIDDEN_DIM'])
    y.append(res_dict['ratio'])
# 对 x,y按x排序
x, y = zip(*sorted(zip(x, y)))
plt.scatter(x, y)
plt.plot([0, 16500], [0.625, 0.625], color='red', linewidth=2.0, linestyle='--')
plt.text(16500, 0.625, 'theoretical limit', ha='right', va='bottom', fontsize=10)
plt.xlabel("HIDDEN_DIM")
plt.ylabel("ratio")
plt.show()
plt.savefig("scatter.png")
# 仿照上面绘制散点图，绘制折线图
x = []
y = []
for res_dict in res:
    x.append(res_dict['HIDDEN_DIM'])
    y.append(res_dict['ratio'])
x, y = zip(*sorted(zip(x, y)))
plt.plot(x, y)
plt.plot([0, 16500], [0.625, 0.625], color='red', linewidth=2.0, linestyle='--')
plt.text(16500, 0.625, 'theoretical limit', ha='right', va='bottom', fontsize=10)
plt.xlabel("HIDDEN_DIM")
plt.ylabel("ratio")
plt.show()
plt.savefig("line.png")

