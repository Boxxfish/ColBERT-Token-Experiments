"""
Performs analysis and experiments related to perturbing MASKs.
"""
import pickle
import numpy as np
import json
from matplotlib import pyplot as plt
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from argparse import ArgumentParser
import pyterrier as pt

from trec_utils import process_ds
if not pt.started():
    pt.init()
from pyterrier_colbert.ranking import ColBERTFactory
import metrics

def main():
    parser = ArgumentParser()
    parser.add_argument("--experiment-adapt-masks", action="store_true")
    parser.add_argument("--visualize-adapt-masks", action="store_true")
    parser.add_argument("--visualize-query-masks", action="store_true")
    parser.add_argument("--visualize-contiguous", action="store_true")
    args = parser.parse_args()
    

    if args.experiment_adapt_masks:
        print("Running MASK adaptation experiment...")

        pytcolbert = ColBERTFactory("http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip", 
                            "./trec_index", "trec", gpu=True)

        topic, qrels = process_ds()

        PAD = 0
        print("Replacing half of the MASKs with PAD, then pruning PAD...")
        dense_e2e_bert_pruned = pytcolbert.end_to_end({PAD}, prune_queries=True, prune_documents=False, remove_masks="half")
        pt.Experiment(
            [dense_e2e_bert_pruned],
            topic,
            qrels,
            filter_by_qrels=True,
            eval_metrics=metrics.eval_metrics,
            save_dir="results",
            save_mode="reuse",
            batch_size=10000,
            verbose=True,
            names=["trec_remove_masks_half"]
        )
        del dense_e2e_bert_pruned
        
        cmp_res = pt.Experiment(
            [None] * 2,
            topic,
            qrels,
            filter_by_qrels=True,
            eval_metrics=metrics.eval_metrics,
            save_dir="results",
            save_mode="reuse",
            # batch_size=5000,
            correction='bonferroni',
            verbose=True,
            baseline=0,
            names=["trec_no_pruning", "trec_remove_masks_half"]
        )

        print(cmp_res)

        try:
            cmp_res.to_csv(f"results/trec_adapt_masks.csv")
        except:
            print("Could not save to csv")

    if args.visualize_adapt_masks:
        # Create ColBERT
        pytcolbert = ColBERTFactory("http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip", 
                            "./trec_index", "trec", gpu=True)
        topic, qrels = process_ds()

        # Create query embeddings for both cases
        query = topic.iloc[20].query
        Q, ids, masks = pytcolbert.args.inference.queryFromText([query], bsize=1, with_ids=True)
        q_embs_normal = Q[0, :, :].cpu()

        PAD, SEP = (0, 102)
        batches = pytcolbert.args.inference.query_tokenizer.tensorize([query], bsize=1)
        for (input_ids, _) in batches:
            q_tok_ids = input_ids[0]
            sep_index = torch.where(q_tok_ids.squeeze() == SEP)[0].item()
            num_masks = 32 - (sep_index + 1)
            masks_to_remove = num_masks // 2 
            toks_in_after = 32 - masks_to_remove
            replace_idxs = list(range(sep_index + 1, toks_in_after))
            
            for (input_ids, _) in batches:
                for i in replace_idxs:
                    input_ids[0][i] = PAD

        with torch.no_grad():
            batchesEmbs = [pytcolbert.args.inference.query(input_ids, attention_mask, to_cpu=False) for input_ids, attention_mask in batches]
            Q, _, masks = (torch.cat(batchesEmbs), torch.cat([ids for ids, _ in batches]), torch.cat([masks for _, masks in batches]))

        q_embs_half = Q[0, :, :].cpu()

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        q_tok_ids = ids[0]
        qtoks = [tok if tok != "[unused0]" else "[Q]" for tok in tokenizer.convert_ids_to_tokens(q_tok_ids)]

        with open("pca_2d.pkl", "rb") as f:
            pca = pickle.load(f)
        xformed_before = pca.transform(q_embs_normal)
        xformed_after = pca.transform(q_embs_half)
        plt.scatter(xformed_before[sep_index + 1:, 0], xformed_before[sep_index + 1:, 1], label="before", alpha=0.5)
        plt.scatter(xformed_after[sep_index + 1:toks_in_after, 0], xformed_after[sep_index + 1:toks_in_after, 1], label="after", alpha=0.5)
        plt.scatter(xformed_before[:sep_index + 1, 0], xformed_before[:sep_index + 1, 1], label="query text tokens", alpha=0.5)
        for i in range(sep_index + 1):
            point = xformed_before[i]
            plt.annotate(qtoks[i], point)
        for i in range(sep_index + 1, 32):
            point = xformed_before[i]
            plt.annotate(i - (sep_index + 1), point)
        for i in range(sep_index + 1, toks_in_after):
            point = xformed_after[i]
            plt.annotate(i - (sep_index + 1), point)
        plt.title(f"Full MASKs vs. Half MASKs:\n\"{query}\"", fontdict={"size": 8})
        plt.legend()
        plt.savefig("full_vs_half_masks.png")

    if args.visualize_query_masks:
        # Create ColBERT
        pytcolbert = ColBERTFactory("http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip", 
                            "./trec_index", "trec", gpu=True)
        topic, qrels = process_ds()

        # Create query embeddings for both cases
        query = topic.iloc[20].query
        Q, ids, masks = pytcolbert.args.inference.queryFromText([query], bsize=1, with_ids=True)
        q_embs_normal = Q[0, :, :].cpu()

        PAD, SEP = (0, 102)
        batches = pytcolbert.args.inference.query_tokenizer.tensorize([query], bsize=1)
        for (input_ids, _) in batches:
            q_tok_ids = input_ids[0]
            sep_index = torch.where(q_tok_ids.squeeze() == SEP)[0].item()
            replace_idxs = list(range(sep_index + 1, 32))
            
            for (input_ids, _) in batches:
                for i in replace_idxs:
                    input_ids[0][i] = PAD

        with torch.no_grad():
            batchesEmbs = [pytcolbert.args.inference.query(input_ids, attention_mask, to_cpu=False) for input_ids, attention_mask in batches]
            Q, q_tok_ids, masks = (torch.cat(batchesEmbs), torch.cat([ids for ids, _ in batches]), torch.cat([masks for _, masks in batches]))

        q_embs_no_mask = Q[0, :, :].cpu()

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        q_tok_ids = ids[0]
        qtoks = [tok if tok != "[unused0]" else "[Q]" for tok in tokenizer.convert_ids_to_tokens(q_tok_ids)]

        with open("pca_2d.pkl", "rb") as f:
            pca = pickle.load(f)
        xformed_before = pca.transform(q_embs_normal)
        xformed_after = pca.transform(q_embs_no_mask)
        plt.scatter(xformed_before[:sep_index + 1, 0], xformed_before[:sep_index + 1, 1], label="before", alpha=0.5)
        plt.scatter(xformed_after[:sep_index + 1, 0], xformed_after[:sep_index + 1, 1], label="after", alpha=0.5)
        for i in range(sep_index + 1):
            point = xformed_before[i]
            plt.annotate(qtoks[i], point)
        plt.title(f"With vs Without MASKs:\n\"{query}\"", fontdict={"size": 8})
        plt.legend()
        plt.savefig("with_vs_without_masks.png")

    if args.visualize_contiguous:
        # Create ColBERT
        pytcolbert = ColBERTFactory("http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip", 
                            "./trec_index", "trec", gpu=True)
        topic, qrels = process_ds()

        # Create query embeddings for all cases
        query = topic.iloc[22].query
        Q, ids, masks = pytcolbert.args.inference.queryFromText([query], bsize=1, with_ids=True)
        q_embs_normal = Q[0, :, :].cpu()

        PAD, SEP, MASK = (0, 102, 103)

        # MASKs before query text
        batches = pytcolbert.args.inference.query_tokenizer.tensorize([query], bsize=1)
        for (input_ids, _) in batches:
            q_tok_ids = input_ids[0]
            sep_index = torch.where(q_tok_ids.squeeze() == SEP)[0].item()
            query_text = input_ids[0][2:sep_index + 1]
            num_masks = 32 - (sep_index + 1)
            input_ids[0][num_masks + 2:] = query_text
            for i in range(2, num_masks + 2):
                input_ids[0][i] = MASK

        with torch.no_grad():
            batchesEmbs = [pytcolbert.args.inference.query(input_ids, attention_mask, to_cpu=False) for input_ids, attention_mask in batches]
            Q, q_tok_ids, masks = (torch.cat(batchesEmbs), torch.cat([ids for ids, _ in batches]), torch.cat([masks for _, masks in batches]))

        q_embs_first_mask = Q[0, :, :].cpu()

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        q_tok_ids = ids[0]
        qtoks = [tok if tok != "[unused0]" else "[Q]" for tok in tokenizer.convert_ids_to_tokens(q_tok_ids)]

        with open("pca_2d.pkl", "rb") as f:
            pca = pickle.load(f)
        xformed_before = pca.transform(q_embs_normal)
        xformed_after = pca.transform(q_embs_first_mask)
        plt.scatter(xformed_before[:sep_index + 1, 0], xformed_before[:sep_index + 1, 1], label="before", alpha=0.5)
        plt.scatter(xformed_after[:2, 0], xformed_after[:2, 1], label="after", alpha=0.5)
        plt.scatter(xformed_after[num_masks + 2:, 0], xformed_after[num_masks + 2:, 1], label="after", alpha=0.5)
        for i in range(sep_index + 1):
            point = xformed_before[i]
            plt.annotate(qtoks[i], point)
        for i in range(2):
            point = xformed_after[i]
            plt.annotate(qtoks[i], point)
        for i in range(2, 32 - num_masks):
            point = xformed_after[num_masks + i]
            plt.annotate(qtoks[i], point)
        plt.title(f"Normal vs MASKs First:\n\"{query}\"", fontdict={"size": 8})
        plt.legend()
        plt.savefig("normal_vs_first_masks.png")

if __name__ == "__main__":
    main()
