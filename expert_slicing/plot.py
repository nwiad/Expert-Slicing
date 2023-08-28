import matplotlib.pyplot as plt
def plot(w, e, tp):
    s = f"_W{w}_E{e}_TP{tp}"
    with open(f"res/results"+s+".txt", "r") as f:
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
    plt.figure()
    plt.scatter(x, y)
    plt.plot([0, 16500], [0.625, 0.625], color='red', linewidth=2.0, linestyle='--')
    plt.text(16500, 0.625, 'theoretical limit', ha='right', va='bottom', fontsize=10)
    plt.xlabel("HIDDEN_DIM")
    plt.ylabel("Ratio of Inference Time")
    plt.savefig("pics/pics"+s+"/scatter"+s+".png")

    plt.figure()
    plt.ylim(0, 0.014)
    plt.xlabel("HIDDEN_DIM")
    plt.ylabel("Inference Time")
    p1 = plt.scatter(x, sliced_time)
    p2 = plt.scatter(x, unsliced_time)
    plt.legend([p1, p2], ["Sliced MoE", "Unsliced MoE"])
    plt.savefig("pics/pics"+s+"/sep_scatter"+s+".png")
    # 折线图
    plt.figure()
    plt.plot(x, y)
    t = (w+1.0)/(2*w)
    plt.plot([0, 16500], [t, t], color='red', linewidth=2.0, linestyle='--')
    plt.text(16500, t, 'theoretical limit', ha='right', va='bottom', fontsize=10)
    plt.xlabel("HIDDEN_DIM")
    plt.ylabel("Ratio of Inference Time")
    plt.savefig("pics/pics"+s+"/line"+s+".png")

    plt.figure()
    plt.ylim(0, 0.014)
    plt.xlabel("HIDDEN_DIM")
    plt.ylabel("Inference Time")
    p1, =plt.plot(x, sliced_time)
    p2, =plt.plot(x, unsliced_time)
    plt.legend([p1, p2], ["Sliced MoE", "Unsliced MoE"])
    plt.savefig("pics/pics"+s+"/sep_line"+s+".png")

plot(2,2,2)
plot(2,4,2)
plot(4,4,2)
plot(4,4,4)