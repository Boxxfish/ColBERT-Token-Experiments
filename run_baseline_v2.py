"""
Runs ColBERTv2 on MS MARCO's dev set.
The expected metrics are MRR@10: 39.7, R@50: 86.8, and R@1K: 98.4.
"""

import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_colbert.ranking import ColBERTFactory
from ir_measures import RR, Recall

eval_metrics = [RR(rel=1), Recall(rel=1)@50, Recall(rel=1)@1000]

pytcolbert = ColBERTFactory("../colbertv2.dnn", "./msmarco_index_v2", "msmarco", gpu=True)
msmarco_ds = pt.get_dataset("irds:msmarco-passage")
topics = msmarco_ds.get_topics("dev")
qrels = msmarco_ds.get_qrels("dev")

cmp_res = pt.Experiment(
    ["baseline_colbertv2_msmarco"],
    topics,
    qrels,
    filter_by_qrels=True,
    eval_metrics=eval_metrics,
    save_dir="results",
    save_mode="reuse",
    correction='bonferroni',
    verbose=True,
    baseline=0,
    names=eval_metrics
)

print(cmp_res)

try:
	cmp_res.to_csv(f"results/baseline_colbertv2_msmarco.csv")
except:
	print("Could not save to csv")

