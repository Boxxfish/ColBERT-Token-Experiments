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

################################
# BERT token pruned calculations
################################

# tok ids for [CLS] and [SEP]
bert_ids = {101, 102}

dense_e2e_bert_pruned = pytcolbert.end_to_end(bert_ids)
print()
print("BERT Tokens Removed Experiment...")
pt.Experiment(
    [dense_e2e_bert_pruned],
    msmarco_ds.get_topics("dev"),
    msmarco_ds.get_qrels("dev"),
    filter_by_qrels=True,
    eval_metrics=["map", RR@10],
    save_dir="results",
    save_mode="reuse",
    batch_size=10000,
    verbose=True,
    names=["bert_tok_removed"]
)

del dense_e2e_bert_pruned

