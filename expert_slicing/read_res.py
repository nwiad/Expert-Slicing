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
plt.scatter(x, y)
plt.xlabel("HIDDEN_DIM")
plt.ylabel("ratio")
plt.show()
plt.savefig("scatter.png")
