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

######################
# Control Calculations
######################
dense_e2e = pytcolbert.end_to_end()

print()
print("No Pruning Experiment...")
pt.Experiment(
    [dense_e2e],
    msmarco_ds.get_topics("dev"),
    msmarco_ds.get_qrels("dev"),
    eval_metrics=["map", RR@10],
    save_dir="results",
    save_mode="reuse",
    batch_size=5000,
    verbose=True,
    names=["no_pruning"]
)

del dense_e2e
