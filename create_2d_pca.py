"""
Creates a 2D PCA model for all doc embeddings.
"""
import numpy as np
from sklearn.decomposition import IncrementalPCA
import pyterrier as pt
if not pt.started():
    pt.init()
from tqdm import tqdm
import pickle
from pathlib import Path

pca = IncrementalPCA(n_components=3, batch_size=10000)
d_embs_path = Path("d_embs")
num_parts = len(list(d_embs_path.iterdir()))
for i in range(num_parts):
    d_embs = np.load(f"d_embs/{i}.npy")
    d_shape = d_embs.shape
    pca.partial_fit(d_embs.reshape(d_shape[0] * d_shape[1], d_shape[2]))
with open("pca.pkl", "wb") as f:
    pickle.dump(pca, f)