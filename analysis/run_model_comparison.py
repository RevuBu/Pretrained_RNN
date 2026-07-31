import numpy as np
from itertools import combinations
from scipy.stats import friedmanchisquare, wilcoxon, kruskal


models = ["SR_Dense", "SR_Moderate", "SR_Sparse",
          "CO_Single", "CO_Multi", "DR_Single", "DR_Multi"]

model_idx = {label: i for i, label in enumerate(
    ["MC_agg", "MM_agg", "SR_Dense", "SR_Moderate", "SR_Sparse",
     "CO_Single", "CO_Multi", "DR_Single", "DR_Multi"])}
file_name = "Data/CO/"
for fname, monkey in [("DSA_C.npy", "Monkey C"), ("DSA_M.npy", "Monkey M")]:
    arr = np.load(file_name+fname)
    print(f"\n{monkey}")

    data = np.array([arr[model_idx[m]] for m in models])
    stat, p = friedmanchisquare(*data)
    print(f"Friedman: chi2={stat:.3f}, p={p:.6f}")

    pairs = []
    for a, b in combinations(range(7), 2):
        w, p_w = wilcoxon(data[a], data[b])
        pairs.append({"m1": models[a], "m2": models[b],
                      "diff": float(np.mean(data[a] - data[b])), "p": p_w})
    pairs.sort(key=lambda x: x["p"])
    for r, pr in enumerate(pairs):
        pr["adj"] = 0.05 / (len(pairs) - r)
        pr["reject"] = pr["p"] < pr["adj"]

    sig = [p for p in pairs if p["reject"]]
    print(f"  Sig: {len(sig)}/{len(pairs)}")
    for pr in sig:
        print(f"    {pr['m1']:15s} vs {pr['m2']:15s}  diff={pr['diff']:+8.6f}  p={pr['p']:.6f}")

    sr = data[[0, 1, 2]].ravel()
    co = data[[3, 4]].ravel()
    dr = data[[5, 6]].ravel()
    h, p_k = kruskal(sr, co, dr)
    print(f"  Kruskal-Wallis (SR vs CO vs DR): H={h:.3f}, p={p_k:.6f}")
