"""
Collects and saves all document embeddings in the index.
"""
import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_colbert.ranking import ColBERTFactory
from tqdm import tqdm
import numpy as np

pytcolbert = ColBERTFactory("http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip", "./msmarco_index", "msmarco", gpu=True)

embs_d_all = []
part_idx = 0
for i in tqdm(range(pytcolbert.numdocs)):
    embs_d_all.append(pytcolbert.rrm.get_embedding(i))
    if (i + 1) % 100000 == 0:
        embs_d_all = np.stack(embs_d_all, 0)
        np.save(f"d_embs/{part_idx}.npy", embs_d_all)
        part_idx += 1
        embs_d_all = []
embs_d_all = np.stack(embs_d_all, 0)
np.save(f"d_embs/{part_idx}.npy", embs_d_all)