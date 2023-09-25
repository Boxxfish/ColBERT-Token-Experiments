"""
Converts processed query data into files for visualization.
"""
from transformers import AutoTokenizer
import json
import numpy as np
from sklearn.decomposition import IncrementalPCA
import pyterrier as pt
if not pt.started():
    pt.init()
from tqdm import tqdm
import pickle
from pathlib import Path

def main():
    with open("q_data.json", "r") as file:
        q_data = json.load(file)
    q_embs = np.load("q_embs.npy")
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    msmarco_ds = pt.get_dataset("msmarco_passage")
    topic = msmarco_ds.get_topics("dev")
    
    # Train PCA
    try:
        with open("pca.pkl", "rb") as f:
            pca = pickle.load(f)
    except:
        print("Couldn't load pickled PCA. Training PCA on all doc embeddings...")
        pca = IncrementalPCA(n_components=3, batch_size=10000)
        d_embs_path = Path("d_embs")
        num_parts = len(list(d_embs_path.iterdir()))
        for i in range(num_parts):
            d_embs = np.load(f"d_embs/{i}.npy")
            d_shape = d_embs.shape
            pca.partial_fit(d_embs.reshape(d_shape[0] * d_shape[1], d_shape[2]))
        with open("pca.pkl", "wb") as f:
            pickle.dump(pca, f)

    # Collect data
    print("Collecting data...")
    all_q_data = []
    for qid in tqdm(q_data):
        query_data = q_data[qid]
        query_text = topic.loc[lambda df: df.qid == qid]["query"].values[0]
        q_tok_data = query_data["q_tok_data"]
        query_tokens = tokenizer.convert_ids_to_tokens([d["tok_id"] for d in q_tok_data])
        query_tokens[1] = "[Q]" # Special case for [Q] token

        # Get query embedding scatter data
        emb_index = query_data["emb_index"]
        query_embs = q_embs[emb_index] # Shape: (32, emb_dim)
        pca_local = IncrementalPCA(n_components=3)
        pca_local.fit(query_embs)
        xformed = pca.transform(query_embs).T # Shape: (3, 32)
        xformed_local = pca_local.transform(query_embs).T # Shape: (3, 32)
        x = xformed[0].tolist()
        y = xformed[1].tolist()
        z = xformed[2].tolist()
        x_local = xformed_local[0].tolist()
        y_local = xformed_local[1].tolist()
        z_local = xformed_local[2].tolist()

        # Get token matching data
        token_matches = []
        for i in range(32):
            q_tok_item = q_tok_data[i]
            d_tok_ids = q_tok_item["d_tok_ids"]
            d_tok_scores = q_tok_item["d_tok_scores"]
            doc_tokens = tokenizer.convert_ids_to_tokens(d_tok_ids)
            doc_tokens = [x if x != "[unused1]" else "[D]" for x in doc_tokens] # Special case for [D] token
            d_tok_ctx = q_tok_item["d_tok_ctx"]
            doc_tokens_reduced = list(set(doc_tokens))
            doc_token_counts = []
            d_tok_ctx_spans = [[t if t != "[unused1]" else "[D]" for t in tokenizer.convert_ids_to_tokens(ctx)] for ctx in d_tok_ctx]
            d_tok_ctx_spans_all = []
            d_tok_scores_all = []
            for d_tok in doc_tokens_reduced:
                doc_token_counts.append(doc_tokens.count(d_tok))
                d_tok_ctx_spans_inner = []
                d_tok_scores_inner = []
                for j, d_tok_inner in enumerate(doc_tokens):
                    if d_tok_inner == d_tok:
                        d_tok_ctx_spans_inner.append(d_tok_ctx_spans[j])
                        d_tok_scores_inner.append(d_tok_scores[j])
                d_tok_ctx_spans_all.append(d_tok_ctx_spans_inner)
                d_tok_scores_all.append(d_tok_scores_inner)
            token_matches.append({
                "doc_tokens": doc_tokens_reduced,
                "doc_token_counts": doc_token_counts,
                "doc_ctx_spans": d_tok_ctx_spans_all,
                "doc_token_scores": d_tok_scores_all,
            })

        q_data_item = {
            "query_text": query_text,
            "query_tokens": query_tokens,
            "scatter_x": x,
            "scatter_y": y,
            "scatter_z": z,
            "scatter_local_x": x_local,
            "scatter_local_y": y_local,
            "scatter_local_z": z_local,
            "token_matches": token_matches,
        }
        all_q_data.append(q_data_item)

    with open("q_data_processed.json", "w") as f:
        json.dump(all_q_data, f)

if __name__ == "__main__":
    main()
