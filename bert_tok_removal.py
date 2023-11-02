# Initialize pyterrier
import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_colbert.ranking import ColBERTFactory
from ir_measures import RR

# Argparse stuff
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--prune-query", action="store_true")
parser.add_argument("--prune-doc", action="store_true")
args = parser.parse_args()
prune_query = args.prune_query
prune_doc = args.prune_doc

# Tokenization stuff
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

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

# Token IDs.
Q = "[unused0]"
D = "[unused1]"
prune_tokens = ["[SEP]"]
prune_set = set(tokenizer.convert_tokens_to_ids(prune_tokens))

dense_e2e_bert_pruned = pytcolbert.end_to_end(prune_set, prune_queries=prune_query, prune_documents=prune_doc)
print(f"Experiment: prune_query={prune_query}, prune_doc={prune_doc}")
pt.Experiment(
    [dense_e2e_bert_pruned],
    msmarco_ds.get_topics("dev"),
    msmarco_ds.get_qrels("dev"),
    filter_by_qrels=True,
    eval_metrics=["map", RR@1, RR@5, RR@10, RR@20],
    save_dir="results",
    save_mode="reuse",
    batch_size=10000,
    verbose=True,
    names=[f"{'_'.join(prune_tokens)}_prune_query_{prune_query}_prune_doc_{prune_doc}"]
)

del dense_e2e_bert_pruned

