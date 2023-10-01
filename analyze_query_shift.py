"""
Performs analysis on queries.
"""
import pickle
import numpy as np
import json
from matplotlib import pyplot as plt
from transformers import AutoTokenizer
from tqdm import tqdm
from argparse import ArgumentParser

def shift_query_data(q_embs_after_current: np.ndarray, sep_idx: int):
    q_embs_after_fixed = q_embs_after_current.copy()
    q_embs_after_fixed[4:sep_idx] = q_embs_after_current[2:sep_idx - 2]
    q_embs_after_fixed[2] = q_embs_after_current[sep_idx - 1]
    q_embs_after_fixed[3] = q_embs_after_current[sep_idx - 2]
    return q_embs_after_fixed

def main():
    parser = ArgumentParser()
    parser.add_argument("--experiment-compare-dists", action="store_true")
    args = parser.parse_args()

    # Load data
    q_embs_before = np.load("shift_artifacts/q_embs_before.npy")
    q_embs_after = np.load("shift_artifacts/q_embs_after.npy")
    all_qids = np.load("shift_artifacts/qids.npy")
    with open("shift_artifacts/swap_metadata.json", "r") as f:
        metadata = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    SEP, MASK = (102, 103)

    if args.experiment_compare_dists:
        print("Running cosine distance experiment...")
        what_is_ids = tokenizer("what is")["input_ids"][1:-1]
        all_dists = []
        for i in tqdm(range(len(metadata))):
            query_data = metadata[i]
            query = query_data["query"]
            qids = all_qids[i]
            sep_idx = np.where(qids == SEP)[0][0]
            
            if not (qids[2] == what_is_ids[0] and qids[3] == what_is_ids[1]):
                continue

            # A valid query is between 3 to 8 tokens, including "what is".
            # Restricting the length prevents very long queries from accidentally drifting semantically.
            if not (sep_idx > (2 + 3) and sep_idx <= (2 + 8)):
                continue

            q_embs_before_current = q_embs_before[i]
            q_embs_after_current = q_embs_after[i]
            q_embs_after_fixed = shift_query_data(q_embs_after_current, sep_idx)

            dists = np.diag(1 - q_embs_before_current @ q_embs_after_fixed.T)
            selected_dists = dists[[0, 1, 2, 4, sep_idx, 12, 31]]
            all_dists.append(selected_dists)
        print("Number of samples:", len(all_dists))
        all_dists = np.stack(all_dists, 0)
        plt.title("Cosine Distance Between Original and Shifted Representation")
        plt.violinplot(all_dists)
        plt.xticks(list(range(1, 8)), ["CLS", "Q", "QUERY:3", "QUERY:5", "SEP", "MASK:13", "MASK:32"])
        plt.xlabel("Token")
        plt.ylabel("Cosine Distance")
        plt.savefig("cosine_dist_experiment.png")
        quit()

    q_idx = 5
    qids = all_qids[q_idx]
    sep_idx = np.where(qids == SEP)[0][0]
    q_embs_before_current = q_embs_before[q_idx][:sep_idx + 1]
    q_embs_after_current = q_embs_after[q_idx][:sep_idx + 1]
    qtoks = [tok if tok != "[unused0]" else "[Q]" for tok in tokenizer.convert_ids_to_tokens(qids)]
    query = metadata[q_idx]["query"]

    # Sanity check - this should go from 0 to 31.
    # sanity_dists = (q_embs_before_current @ q_embs_before_current.T).argmax(1)
    # print(sanity_dists)
    # sanity_dists = (q_embs_after_current @ q_embs_after_current.T).argmax(1)
    # print(sanity_dists)
    
    q_embs_after_fixed = shift_query_data(q_embs_after_current, sep_idx)
    shifted_qids = qids.copy()
    shifted_qids[2:sep_idx - 2] = qids[4:sep_idx]
    shifted_qids[sep_idx - 1] = qids[2]
    shifted_qids[sep_idx - 2] = qids[3]
    shifted_query = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(shifted_qids[2:sep_idx]))
    
    # Another sanity check - the query IDs should be swapped correctly.
    # print(qids)
    # qids_temp = qids.copy()
    # qids_temp[4:sep_idx] = qids[2:sep_idx - 2]
    # qids_temp[2] = qids[sep_idx - 1]
    # qids_temp[3] = qids[sep_idx - 2]
    # print(qids_temp)

    with open("pca_2d.pkl", "rb") as f:
        pca = pickle.load(f)
    pca.fit(np.concatenate([q_embs_before_current, q_embs_after_current], 0))
    xformed_before = pca.transform(q_embs_before_current)[:, :2]
    xformed_after = pca.transform(q_embs_after_fixed)[:, :2]
    plt.scatter(xformed_before[:, 0], xformed_before[:, 1], label="before")
    plt.scatter(xformed_after[:, 0], xformed_after[:, 1], label="after")
    for i, point in enumerate(xformed_before):
        plt.annotate(qtoks[i], point)
    for i, point in enumerate(xformed_after):
        plt.annotate(qtoks[i], point)
    plt.title(f"\"{query}\" vs. \"{shifted_query}\"", fontdict={"size": 8})
    plt.legend()
    plt.savefig("shift.png")

    target_token_idxs = [2, sep_idx - 1]
    check_token_idxs = [0, 1, sep_idx]
    old_dists = q_embs_before_current @ q_embs_before_current.T
    new_dists = q_embs_before_current @ q_embs_after_fixed.T
    print("Query:", query)
    for target_token_idx in target_token_idxs:
        for check_token_idx in check_token_idxs:
            old_dist = old_dists[check_token_idx, target_token_idx]
            new_dist = new_dists[check_token_idx, target_token_idx]
            print("Checking token", check_token_idx, "against", target_token_idx, ", shifted by", new_dist - old_dist)

if __name__ == "__main__":
    main()
