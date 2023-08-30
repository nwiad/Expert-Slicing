import matplotlib.pyplot as plt

B = 8
SQ = 1024

def plot(w, e, tp):
    s = f"_W{w}_E{e}_TP{tp}"
    with open(f"res_B{B}_SQ{SQ}/results"+s+".txt", "r") as f:
        print(f.name)
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
    plt.plot([0, 16500], [1, 1], color='red', linewidth=2.0, linestyle='--')
    plt.text(16500, 1, 'Threshold Value (Ratio=1.0)', ha='right', va='bottom', fontsize=10)
    t = 1.0 / tp
    plt.plot([0, 16500], [t, t], color='red', linewidth=2.0, linestyle='--')
    plt.text(16500, t, 'Theoretical Limit', ha='right', va='bottom', fontsize=10)
    plt.xlabel("HIDDEN_DIM")
    plt.ylabel("Ratio of Inference Time")
    plt.title(f"B={B}, SQ={SQ}, E={e}, TP={tp}")
    plt.savefig(f"pics_B{B}_SQ{SQ}/pics"+s+"/scatter"+s+".png")

    plt.figure()
    plt.xlabel("HIDDEN_DIM")
    plt.ylabel("Inference Time")
    p1 = plt.scatter(x, sliced_time)
    p2 = plt.scatter(x, unsliced_time)
    plt.legend([p1, p2], ["Sliced MoE", "Unsliced MoE"])
    plt.title(f"B={B}, SQ={SQ}, E={e}, TP={tp}")
    plt.savefig(f"pics_B{B}_SQ{SQ}/pics"+s+"/sep_scatter"+s+".png")
    # 折线图
    plt.figure()
    plt.plot(x, y)
    plt.plot([0, 16500], [1, 1], color='red', linewidth=2.0, linestyle='--')
    plt.text(16500, 1, 'Threshold Value (Ratio=1.0)', ha='right', va='bottom', fontsize=10)
    t = 1.0 /tp
    plt.plot([0, 16500], [t, t], color='red', linewidth=2.0, linestyle='--')
    plt.text(16500, t, 'Theoretical Limit', ha='right', va='bottom', fontsize=10)
    plt.xlabel("HIDDEN_DIM")
    plt.ylabel("Ratio of Inference Time")
    plt.title(f"B={B}, SQ={SQ}, E={e}, TP={tp}")
    plt.savefig(f"pics_B{B}_SQ{SQ}/pics"+s+"/line"+s+".png")

    plt.figure()
    plt.xlabel("HIDDEN_DIM")
    plt.ylabel("Inference Time")
    p1, =plt.plot(x, sliced_time)
    p2, =plt.plot(x, unsliced_time)
    plt.legend([p1, p2], ["Sliced MoE", "Unsliced MoE"])
    plt.title(f"B={B}, SQ={SQ}, E={e}, TP={tp}")
    plt.savefig(f"pics_B{B}_SQ{SQ}/pics"+s+"/sep_line"+s+".png")

def all_plot(w, e, tp):
    s = f"_W{w}_E{e}_TP{tp}"
    with open(f"res_B{B}_SQ{SQ}/results"+s+".txt", "r") as f:
        print(f.name)
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
    # 折线图
    plt.figure()
    p, = plt.plot(x, y)
    plt.plot([0, 16500], [1, 1], color='red', linewidth=2.0, linestyle='--')
    plt.text(16500, 1, 'Ratio=1.0', ha='right', va='bottom', fontsize=10)
    t = 1.0 /tp
    plt.plot([0, 16500], [t, t], color='red', linewidth=2.0, linestyle='--')
    plt.text(16500, t, 'Ratio Theoretical Limit', ha='right', va='bottom', fontsize=10)
    plt.xlabel("HIDDEN_DIM")
    plt.ylabel("Ratio of Inference Time")
    # 把y轴设在左边
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.set_ylabel("Inference Time")
    p1, = ax2.plot(x, sliced_time, color='blue')
    p2, = ax2.plot(x, unsliced_time, color='orange')
    plt.legend([p1, p2, p], ["Sliced MoE", "Unsliced MoE", "Ratio of Inference Time"])
    # 添加标题： title
    plt.title(f"B={B}, SQ={SQ}, E={e}, TP={tp}")
    plt.savefig(f"pics_B{B}_SQ{SQ}/pics"+s+"/all_line"+s+".png")

print(f"Plotting B={B} SQ={SQ}")
args =[(2,2,2), (2,4,2), (4,4,2), (4,4,4)]
for arg in args:
    try:
        all_plot(*arg)
        plot(*arg)
    except Exception as e:
        print(f"Failed for {arg}: {e}")
        continue