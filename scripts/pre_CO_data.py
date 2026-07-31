import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.metrics import auc


def cca_analysis(fr1, fr2, dim):
    cond, time, _ = fr1.shape
    fr1_flatten = np.reshape(fr1, (cond * time, fr1.shape[2]))
    pca = PCA(n_components=10)
    X = pca.fit_transform(fr1_flatten)
    fr2_flatten = np.reshape(fr2, (cond * time, fr2.shape[2]))
    pca = PCA(n_components=10)
    Y = pca.fit_transform(fr2_flatten)
    cca = CCA(n_components=dim)
    cca.fit(X, Y)
    X_c, Y_c = cca.transform(X, Y)
    r = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(X_c.shape[1])]
    return r


# Load data
fr_MC = np.load("Data/CO/fr_MC_5.npy")[:, 20:-20]
fr_MM = np.load("Data/CO/fr_MM_5.npy")[:, 20:-20]
fr_SR_Dense = np.load("Data/CO/fr_SR_200.npy")[:, 20:-20]
fr_SR_Moderate = np.load("Data/CO/fr_SR_Dense.npy")[:, 20:-20]
fr_SR_Sparse = np.load("Data/CO/fr_SR_50.npy")[:, 20:-20]
fr_CO_Single = np.load("Data/CO/fr_CO_Single.npy")[:, 20:-20]
fr_CO_Multi = np.load("Data/CO/fr_CO_Multi.npy")[:, 20:-20]
fr_DR_Single = np.load("Data/CO/fr_DR_Single.npy")[:, 20:-20]
fr_DR_Multi = np.load("Data/CO/fr_DR_Multi.npy")[:, 20:-20]

pc = np.arange(1, 11)
datasets = [
    ("Monkey C", fr_MC),
    ("Monkey M", fr_MM),
    ("SR_Dense", fr_SR_Dense),
    ("SR_Moderate", fr_SR_Moderate),
    ("SR_Sparse", fr_SR_Sparse),
    ("CO_Single", fr_CO_Single),
    ("CO_Multi", fr_CO_Multi),
    ("DR_Single", fr_DR_Single),
    ("DR_Multi", fr_DR_Multi),
]

# Monkey C analysis
CCA_C, AUC_C = {}, {}
for i in range(4):
    fr_MC_i = np.load(f"Data/CO/fr_MC_{i+1}.npy")[:, 20:-20]
    for name, fr in datasets:
        key = f"vs {name}"
        r = cca_analysis(fr_MC_i, fr, 10)
        if key not in CCA_C:
            CCA_C[key] = []
            AUC_C[key] = []
        CCA_C[key].append(r)
        AUC_C[key].append(auc(pc, r))

for k in CCA_C:
    CCA_C[k] = np.array(CCA_C[k])
    AUC_C[k] = np.array(AUC_C[k])

with open("Data/CO/CCA_C.pkl", "wb") as f:
    pickle.dump(CCA_C, f)
with open("Data/CO/AUC_C.pkl", "wb") as f:
    pickle.dump(AUC_C, f)

# Monkey M analysis
CCA_M, AUC_M = {}, {}
for i in range(4):
    fr_MM_i = np.load(f"Data/CO/fr_MM_{i+1}.npy")[:, 20:-20]
    for name, fr in datasets:
        key = f"vs {name}"
        r = cca_analysis(fr_MM_i, fr, 10)
        if key not in CCA_M:
            CCA_M[key] = []
            AUC_M[key] = []
        CCA_M[key].append(r)
        AUC_M[key].append(auc(pc, r))

for k in CCA_M:
    CCA_M[k] = np.array(CCA_M[k])
    AUC_M[k] = np.array(AUC_M[k])

with open("Data/CO/CCA_M.pkl", "wb") as f:
    pickle.dump(CCA_M, f)
with open("Data/CO/AUC_M.pkl", "wb") as f:
    pickle.dump(AUC_M, f)
