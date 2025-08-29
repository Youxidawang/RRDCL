import numpy as np
import os
def avg_result():
    with open("../code/0.txt", "r", encoding='utf-8') as f:  # 打开文本
        data = f.read()
        list = data.split()
        f1 = []
        p = []
        r = []
        for i in range(1, len(list), 6):
            num = list[i]
            num = num[:-1]
            num = float(num)
            f1.append(num)
        for i in range(3, len(list), 6):
            num = list[i]
            num = num[:-1]
            num = float(num)
            p.append(num)
        for i in range(5, len(list), 6):
            num = list[i]
            num = num[:-1]
            num = float(num)
            r.append(num)
        avg_f1 = np.mean(f1)
        avg_p = np.mean(p)
        avg_r = np.mean(r)
    return avg_f1, avg_p, avg_r
