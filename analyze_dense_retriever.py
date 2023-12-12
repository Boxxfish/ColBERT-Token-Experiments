"""
Computes and analyzes the results of using CLS and SEP as dense retrievers.
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
    parser.add_argument("--experiment", action="store_true")
    args = parser.parse_args()

    if args.experiment:
        pytcolbert = ColBERTFactory("http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip", 
                            "./trec_index", "trec", gpu=True)
        topic, qrels = process_ds()

        # Condition 1: Just first query token
        dense_e2e_bert_pruned = pytcolbert.end_to_end(set(), prune_queries=False, prune_documents=False, keep_pos=2)
        print("Using just first token...")
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
            names=["trec_just_first"]
        )
        del dense_e2e_bert_pruned
        
        # # Condition 2: Just CLS
        CLS = 101
        dense_e2e_bert_pruned = pytcolbert.end_to_end(set(), prune_queries=False, prune_documents=False, keep_tok=CLS)
        print("Using just CLS...")
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
            names=["trec_just_cls"]
        )
        del dense_e2e_bert_pruned

        # # Condition 3: Just SEP
        SEP = 102
        dense_e2e_bert_pruned = pytcolbert.end_to_end(set(), prune_queries=False, prune_documents=False, keep_tok=SEP)
        print("Using just SEP...")
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
            names=["trec_just_sep"]
        )
        del dense_e2e_bert_pruned

        # Get results
        cmp_res = pt.Experiment(
            [None] * 3,
            topic,
            qrels,
            filter_by_qrels=True,
            eval_metrics=[MAP, RR@10, NDCG@10, NDCG@1000],
            save_dir="results",
            save_mode="reuse",
            # batch_size=5000,
            correction='bonferroni',
            verbose=True,
            baseline=0,
            names=["trec_just_first", "trec_just_cls", "trec_just_sep"]
        )
        
        print(cmp_res)

        try:
            cmp_res.to_csv(f"results/trec_dense_retriever.csv")
        except:
            print("Could not save to csv")

if __name__ == "__main__":
    main()