# Initialize pyterrier
import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_colbert.ranking import ColBERTFactory
from ir_measures import RR

#create a ColBERT ranking factory based on the pretrained checkpoint
pytcolbert = ColBERTFactory("http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip", 
                            "./msmarco_index", "msmarco", gpu=True)

# Download and initialize the msmarco dataset
msmarco_ds = pt.get_dataset("msmarco_passage")

# Get list of usable queries that have a corresponding relevant in our limited index
qids = msmarco_ds.get_qrels("dev")

# comparisons = [
# 	["no_pruning", "[CLS]_[SEP]_[MASK]_prune_query_True_prune_doc_False"],
# 	["no_pruning", "[CLS]_[SEP]_[MASK]_prune_query_True_prune_doc_False"],
# ]

cmp_names = [
	"no_pruning", 
    "[CLS]_[SEP]_[MASK]_prune_query_False_prune_doc_False", 
    '[CLS]_[SEP]_[MASK]_prune_query_True_prune_doc_False', 
    '[CLS]_[SEP]_[MASK]_prune_query_True_prune_doc_True'
]

cmp_res = pt.Experiment(
    [None] * len(cmp_names),
    msmarco_ds.get_topics("dev"),
    msmarco_ds.get_qrels("dev"),
    filter_by_qrels=True,
    eval_metrics=["map", RR@1, RR@5, RR@10, RR@20],
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
	cmp_res.to_csv(f"results/[CLS]_[SEP]_[MASK]-all_cmp.csv")
except:
	print("Could not save to csv")

