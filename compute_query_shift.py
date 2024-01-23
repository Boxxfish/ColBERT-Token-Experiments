"""
Computes a query's representation before and after shifting the first two words to the end.
"""
import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_colbert.ranking import ColBERTFactory
from tqdm import tqdm
import numpy as np
import torch
import json

from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument("--v2", action="store_true")
    args = parser.parse_args()

    # Boilerplate loading stuff
    if not args.v2:
        pytcolbert = ColBERTFactory("http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip", "./msmarco_index", "msmarco", gpu=True)
    else:
        pytcolbert = ColBERTFactory("../colbertv2.dnn", "./msmarco_index_v2", "msmarco", gpu=True)

    msmarco_ds = pt.get_dataset("msmarco_passage")
    topic = msmarco_ds.get_topics("dev")
    SEP = 102

    all_q_embs_before = []
    all_q_embs_after = []
    all_qids = []
    metadata = []
    for i in tqdm(range(len(topic))):
        # Compute query representations
        row = topic.iloc[i]
        query: str = row.query
        q_parts = query.split()
        if len(q_parts) < 2:
            continue
        swapped_query = " ".join(q_parts[2:] + [q_parts[1], q_parts[0]])
        q_embs_before, q_ids_before, q_mask_before = pytcolbert.args.inference.queryFromText([query], with_ids=True)
        q_ids_swapped = q_ids_before.clone()
        sep_idx = torch.where(q_ids_before.squeeze() == SEP)[0].item()
        q_ids_swapped[0, 2:(sep_idx - 2)] = q_ids_before[0, 4:sep_idx]
        q_ids_swapped[0, sep_idx - 1] = q_ids_before[0, 2]
        q_ids_swapped[0, sep_idx - 2] = q_ids_before[0, 3]
        q_embs_after = pytcolbert.args.inference.query(q_ids_swapped, q_mask_before)

        # Add to main lists
        all_q_embs_before.append(q_embs_before.cpu().numpy().squeeze())
        all_q_embs_after.append(q_embs_after.cpu().numpy().squeeze())
        all_qids.append(q_ids_before.squeeze())
        metadata.append({
            "query": query,
            "swapped_query": swapped_query,
        })
    
    # Save arrays
    file_suffix = "_v2" if args.v2 else ""
    np.save(f"shift_artifacts/q_embs_before{file_suffix}.npy", np.stack(all_q_embs_before))
    np.save(f"shift_artifacts/q_embs_after{file_suffix}.npy", np.stack(all_q_embs_after))
    np.save(f"shift_artifacts/qids{file_suffix}.npy", np.stack(all_qids))
    with open(f"shift_artifacts/swap_metadata{file_suffix}.json", "w") as f:
        json.dump(metadata, f)

if __name__ == "__main__":
    main()