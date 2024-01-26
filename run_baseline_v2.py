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

#pytcolbert = ColBERTFactory("../colbertv2.dnn", "./msmarco_index_v2", "msmarco", gpu=False)
msmarco_ds = pt.get_dataset("msmarco_passage")
topics = msmarco_ds.get_topics("dev")
qrels = msmarco_ds.get_qrels("dev")

#e2e = pytcolbert.end_to_end(
#    set(),
#    prune_queries=False,
#    prune_documents=False,
#)

cmp_res = pt.Experiment(
    [None],
    topics,
    qrels,
    filter_by_qrels=True,
    eval_metrics=eval_metrics,
    save_dir="results",
    save_mode="reuse",
    verbose=True,
    baseline=0,
    names=["baseline_colbertv2_msmarco"]
)

print(cmp_res)

try:
	cmp_res.to_csv(f"results/baseline_colbertv2_msmarco.csv")
except:
	print("Could not save to csv")

