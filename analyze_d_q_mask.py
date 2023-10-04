"""
Computes and analyzes the results of substituting [Q] with [D], [MASK], or [PAD].
"""

from argparse import ArgumentParser
import pickle
from matplotlib import pyplot as plt
import pyterrier as pt

from trec_utils import process_ds
if not pt.started():
    pt.init()
from pyterrier_colbert.ranking import ColBERTFactory
from ir_measures import MAP, NDCG, RR
from tqdm import tqdm
import numpy as np
import json

def main():
    parser = ArgumentParser()
    parser.add_argument("--compute-embeddings", action="store_true")
    parser.add_argument("--analyze")
    parser.add_argument("--experiment", action="store_true")
    args = parser.parse_args()

    if args.compute_embeddings:
        # Boilerplate loading stuff
        pytcolbert = ColBERTFactory("http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip", "./msmarco_index", "msmarco", gpu=True)
        msmarco_ds = pt.get_dataset("msmarco_passage")
        topic = msmarco_ds.get_topics("dev")
        MASK, PAD, D = (103, 0, 2)

        all_q_embs_q = []
        all_q_embs_d = []
        all_q_embs_mask = []
        all_q_embs_pad = []
        all_qids = []
        metadata = []
        for i in tqdm(range(len(topic))):
            # Compute query representations
            row = topic.iloc[i]
            query: str = row.query
            q_embs_q, q_ids_q, q_mask_q = pytcolbert.args.inference.queryFromText([query], with_ids=True)
            q_ids_q = q_ids_q.squeeze()
            q_ids_d = q_ids_q.clone()
            q_ids_d[1] = D
            q_ids_mask = q_ids_q.clone()
            q_ids_mask[1] = MASK
            q_ids_pad = q_ids_q.clone()
            q_ids_pad[1] = PAD
            q_embs_d = pytcolbert.args.inference.query(q_ids_d.unsqueeze(0), q_mask_q)
            q_embs_mask = pytcolbert.args.inference.query(q_ids_mask.unsqueeze(0), q_mask_q)
            q_embs_pad = pytcolbert.args.inference.query(q_ids_pad.unsqueeze(0), q_mask_q)

            # Add to main lists
            all_q_embs_q.append(q_embs_q.squeeze().cpu().numpy())
            all_q_embs_d.append(q_embs_d.squeeze().cpu().numpy())
            all_q_embs_mask.append(q_embs_mask.squeeze().cpu().numpy())
            all_q_embs_pad.append(q_embs_pad.squeeze().cpu().numpy())
            all_qids.append(q_ids_q.cpu().numpy())
            metadata.append({
                "query": query,
            })
        
        # Save arrays
        np.save("d_q_mask_artifacts/q_embs_q.npy", np.stack(all_q_embs_q))
        np.save("d_q_mask_artifacts/q_embs_d.npy", np.stack(all_q_embs_d))
        np.save("d_q_mask_artifacts/q_embs_mask.npy", np.stack(all_q_embs_mask))
        np.save("d_q_mask_artifacts/q_embs_pad.npy", np.stack(all_q_embs_pad))
        np.save("d_q_mask_artifacts/qids.npy", np.stack(all_qids))
        with open("d_q_mask_artifacts/metadata.json", "w") as f:
            json.dump(metadata, f)

    if args.analyze:
        # Load data
        q_embs_q = np.load("d_q_mask_artifacts/q_embs_q.npy")
        q_embs_d = np.load("d_q_mask_artifacts/q_embs_d.npy")
        q_embs_mask = np.load("d_q_mask_artifacts/q_embs_mask.npy")
        q_embs_pad = np.load("d_q_mask_artifacts/q_embs_pad.npy")
        with open("d_q_mask_artifacts/metadata.json", "r") as f:
            metadata = json.load(f)

        q_id = 107
        query = metadata[q_id]["query"]
        q_embs_q = q_embs_q[q_id]
        q_embs_d = q_embs_d[q_id]
        q_embs_mask = q_embs_mask[q_id]
        q_embs_pad = q_embs_pad[q_id]
        with open("pca_2d.pkl", "rb") as f:
            pca = pickle.load(f)
        xformed_q = pca.transform(q_embs_q)
        xformed_d = pca.transform(q_embs_d)
        xformed_mask = pca.transform(q_embs_mask)
        xformed_pad = pca.transform(q_embs_pad)
        plt.scatter(xformed_q[:, 0], xformed_q[:, 1], label="Q")
        for i, point in enumerate(xformed_q):
            plt.annotate(i + 1, point)
        if args.analyze == "d":
            plt.scatter(xformed_d[:, 0], xformed_d[:, 1], label="D")
            for i, point in enumerate(xformed_d):
                plt.annotate(i + 1, point)
        if args.analyze == "mask":
            plt.scatter(xformed_mask[:, 0], xformed_mask[:, 1], label="MASK")
            for i, point in enumerate(xformed_mask):
                plt.annotate(i + 1, point)
        if args.analyze == "pad":
            plt.scatter(xformed_pad[:, 0], xformed_pad[:, 1], label="PAD (None)")
            for i, point in enumerate(xformed_pad):
                plt.annotate(i + 1, point)
        plt.title(query, fontdict={"size": 8})
        plt.legend()
        plt.savefig(f"scatter_{args.analyze}.png")

    if args.experiment:
        pytcolbert = ColBERTFactory("http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip", 
                            "./trec_index", "trec", gpu=True)

        topic, qrels = process_ds()

        # Condition 1: Control
        # dense_e2e_bert_pruned = pytcolbert.end_to_end(set(), prune_queries=False, prune_documents=False)
        # print("Using Q (Control)...")
        # pt.Experiment(
        #     [dense_e2e_bert_pruned],
        #     topic,
        #     qrels,
        #     filter_by_qrels=True,
        #     eval_metrics=[MAP, RR@10, NDCG@10, NDCG@1000],
        #     save_dir="results",
        #     save_mode="reuse",
        #     batch_size=10000,
        #     verbose=True,
        #     names=["trec_q"]
        # )
        # del dense_e2e_bert_pruned
        
        # Condition 2: D
        D = 2
        # dense_e2e_bert_pruned = pytcolbert.end_to_end(set(), prune_queries=False, prune_documents=False, replace_q=D)
        # print("Using D...")
        # pt.Experiment(
        #     [dense_e2e_bert_pruned],
        #     topic,
        #     qrels,
        #     filter_by_qrels=True,
        #     eval_metrics=[MAP, RR@10, NDCG@10, NDCG@1000],
        #     save_dir="results",
        #     save_mode="reuse",
        #     batch_size=10000,
        #     verbose=True,
        #     names=["trec_d"]
        # )
        # del dense_e2e_bert_pruned
        
        # Condition 3: PAD
        # PAD = 0
        # dense_e2e_bert_pruned = pytcolbert.end_to_end({PAD}, prune_queries=True, prune_documents=False, replace_q=PAD)
        # print("Using PAD, then removing PAD...")
        # pt.Experiment(
        #     [dense_e2e_bert_pruned],
        #     topic,
        #     qrels,
        #     filter_by_qrels=True,
        #     eval_metrics=[MAP, RR@10, NDCG@10, NDCG@1000],
        #     save_dir="results",
        #     save_mode="reuse",
        #     batch_size=10000,
        #     verbose=True,
        #     names=["trec_pad"]
        # )
        # del dense_e2e_bert_pruned

        # Condition 4: D, but D is removed afterwards
        D = 2
        dense_e2e_bert_pruned = pytcolbert.end_to_end({D}, prune_queries=True, prune_documents=False, replace_q=D)
        print("Using D, then removing D...")
        pt.Experiment(
            [dense_e2e_bert_pruned],
            topic,
            qrels,
            filter_by_qrels=True,
            eval_metrics=[MAP, RR@10, NDCG@10, NDCG@1000],
            save_dir="results",
            save_mode="reuse",
            batch_size=10000,
            verbose=True,
            names=["trec_d_removed"]
        )
        del dense_e2e_bert_pruned

        # Condition 6: Q, but Q is removed afterwards
        Q = 1
        dense_e2e_bert_pruned = pytcolbert.end_to_end({Q}, prune_queries=True, prune_documents=False)
        print("Using Q, then removing Q...")
        pt.Experiment(
            [dense_e2e_bert_pruned],
            topic,
            qrels,
            filter_by_qrels=True,
            eval_metrics=[MAP, RR@10, NDCG@10, NDCG@1000],
            save_dir="results",
            save_mode="reuse",
            batch_size=10000,
            verbose=True,
            names=["trec_q_removed"]
        )
        del dense_e2e_bert_pruned

        # Get results
        cmp_res = pt.Experiment(
            [None] * 2,
            topic,
            qrels,
            filter_by_qrels=True,
            eval_metrics=[MAP, RR@10, NDCG@10, NDCG@1000],
            save_dir="results",
            save_mode="reuse",
            # batch_size=5000,
            # correction='bonferroni',
            verbose=True,
            baseline=0,
            names=["trec_q", "trec_d"]
        )

        
        print(cmp_res)

        try:
            cmp_res.to_csv(f"results/trec_q_d_results.csv")
        except:
            print("Could not save to csv")

        # Experiment 2: Q (removed) vs D (removed) vs PAD
        cmp_res = pt.Experiment(
            [None] * 3,
            topic,
            qrels,
            filter_by_qrels=True,
            eval_metrics=[MAP, RR@10, NDCG@10, NDCG@1000],
            save_dir="results",
            save_mode="reuse",
            # batch_size=5000,
            # correction='bonferroni',
            verbose=True,
            baseline=0,
            names=["trec_q_removed", "trec_d_removed", "trec_pad"]
        )

        
        print(cmp_res)

        try:
            cmp_res.to_csv(f"results/trec_q_d_pad_removed_results.csv")
        except:
            print("Could not save to csv")

if __name__ == "__main__":
    main()