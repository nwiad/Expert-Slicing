with open("results.txt", "r") as f:
    res = []
    while line:=f.readline():
        res.append(eval(line))
    print(res)
for res_dict in res:
    print(res_dict['HIDEN_DIM'], res_dict['ratio'])