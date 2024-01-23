import pyterrier as pt
from mod_utils import load_colbert
if not pt.started():
    pt.init()
from trec_utils import process_ds
import metrics

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--v2")
args = parser.parse_args()
suffix = "_v2" if args.v2 else ""

pytcolbert = load_colbert("trec", args.v2)

topics, qrels = process_ds()

cmp_names = [
	f"trec_baseline{suffix}",
	f"trec_remap_special_toks{suffix}",
	f"trec_remap_masks{suffix}",
    f"trec_remap_masks_to_terms{suffix}",
]

cmp_res = pt.Experiment(
    [None] * len(cmp_names),
    topics,
    qrels,
    filter_by_qrels=True,
    eval_metrics=metrics.eval_metrics,
    save_dir="results",
    save_mode="reuse",
    correction='bonferroni',
    verbose=True,
    baseline=0,
    names=cmp_names
)

print(cmp_res)

try:
	cmp_res.to_csv(f"results/trec_cluster_results.csv")
except:
	print("Could not save to csv")

