# Initialize pyterrier
import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_colbert.ranking import ColBERTFactory
from ir_measures import MAP, RR, NDCG
from trec_utils import process_ds
import metrics

#create a ColBERT ranking factory based on the pretrained checkpoint
pytcolbert = ColBERTFactory("http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip", 
                            "./trec_index", "trec", gpu=True)

topics, qrels = process_ds()

cmp_names = [
	"trec_no_pruning",
	"trec_remap_special_toks",
	"trec_remap_masks",
        "trec_remap_masks_to_terms",
]

cmp_res = pt.Experiment(
    [None] * len(cmp_names),
    topics,
    qrels,
    filter_by_qrels=True,
    eval_metrics=metrics.eval_metrics,
    save_dir="results",
    save_mode="reuse",
    # batch_size=5000,
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

