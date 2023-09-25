"""
For a given number of queries...
 1. Uses BM25 to collect the top-k number of relevant documents.
 2. For each query token...
     a. Finds the doc token selected and MaxSim score.
     b. Saves the contextualized embedding.

This generates the following:
 - q_embs.npy: Embeddings for each query term. Shape: (q_count, 32, emb_dim)
 - q_data.json: Data for the search process. A dict where for each q_id as key,
   the query embedding index and matched doc token ids + MaxSim score is saved.
 - d_embs.npy: Embeddings for each of the documents. Shape: (q_count, k, 180, emb_dim)
"""
from argparse import ArgumentParser
import numpy as np
import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_colbert.ranking import ColBERTFactory
import torch
import json
from tqdm import tqdm

def main():
    # Parse args
    parser = ArgumentParser()
    parser.add_argument("--queries", default=10)
    parser.add_argument("--k", default=1000)
    args = parser.parse_args()

    # ColBERT/Terrier boilerplate
    pytcolbert = ColBERTFactory("http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip", "./msmarco_index", "msmarco", gpu=True)
    dense_e2e = pytcolbert.end_to_end() % int(args.k)

    # Save all doc embeddings in batches.
    # Only needs to be done once, if you need to do this again, uncomment this
    # section.
    
    # embs_d_all = []
    # part_idx = 0
    # for i in tqdm(range(pytcolbert.numdocs)):
    #     embs_d_all.append(pytcolbert.rrm.get_embedding(i))
    #     if (i + 1) % 100000 == 0:
    #         embs_d_all = np.stack(embs_d_all, 0)
    #         np.save(f"d_embs/{part_idx}.npy", embs_d_all)
    #         part_idx += 1
    #         embs_d_all = []
    # embs_d_all = np.stack(embs_d_all, 0)
    # np.save(f"d_embs/{part_idx}.npy", embs_d_all)

    # Retrieve and process top docs for each query
    msmarco_ds = pt.get_dataset("msmarco_passage")
    embs = []
    q_data_all = {}
    print("Processing queries...")
    for i in tqdm(range(int(args.queries))):
        row = msmarco_ds.get_topics("dev").iloc[i]
        qid = row.qid
        query = row.query
        q_results = dense_e2e.search(query)
        q_tok_data = [{"tok_id": None, "d_tok_ids": [], "d_tok_scores": [], "d_tok_ctx": []} for _ in range(32)]
        embs_d_for_query = []
        for doc_id in q_results["docid"]:
            ids_q, ids_d, embs_q, embs_d, interaction = pytcolbert.explain_doc(query, doc_id)
            interaction = interaction.T # Shape: (32, 180)
            max_sim_ids = np.argmax(interaction[:, :ids_d.shape[0]], axis=1)  # Shape: (32)
            ids_d_max = ids_d[max_sim_ids]
            max_sim_max = np.max(interaction, axis=1)
            for j in range(32):
                q_tok_data[j]["tok_id"] = int(ids_q.squeeze(0)[j])
                q_tok_data[j]["d_tok_ids"].append(int(ids_d_max[j]))
                q_tok_data[j]["d_tok_scores"].append(float(max_sim_max[j]))
                ctx = []
                ctx_len = 7
                d_tok_idx = max_sim_ids[j]
                min_idx = max(-ctx_len // 2 + d_tok_idx + 1, 0)
                max_idx = min(min_idx + ctx_len, ids_d.shape[0])
                for tok_idx in range(min_idx, max_idx):
                    ctx.append(int(ids_d[tok_idx]))
                q_tok_data[j]["d_tok_ctx"].append(ctx)
        embs.append(embs_q.squeeze(0).cpu().numpy())
        q_data = {
            "q_tok_data": q_tok_data,
            "emb_index": i,
        }
        q_data_all[qid] = q_data
    
    # Save data
    embs = np.stack(embs, 0)
    np.save("q_embs.npy", embs)

    with open("q_data.json", "w") as f:
        json.dump(q_data_all, f)

if __name__ == "__main__":
    main()
