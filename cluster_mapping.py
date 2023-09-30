# Initialize pyterrier
import pandas as pd
import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_colbert.ranking import ColBERTFactory
from ir_measures import RR, NDCG, MAP
from nltk.corpus import stopwords
from trec_utils import process_ds

# Argparse stuff
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--remap-special-toks", action="store_true")
parser.add_argument("--remap-masks", action="store_true")
args = parser.parse_args()

# Tokenization stuff
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

#create a ColBERT ranking factory based on the pretrained checkpoint
pytcolbert = ColBERTFactory("http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip", 
                            "./trec_index", "trec", gpu=True)

topic, qrels = process_ds()

######################
# Control Calculations
######################
# dense_e2e = pytcolbert.end_to_end()

# print()
# print("No Pruning Experiment...")
# pt.Experiment(
#     [dense_e2e],
#     msmarco_ds.get_topics("dev"),
#     msmarco_ds.get_qrels("dev"),
#     eval_metrics=["map", RR@10],
#     save_dir="results",
#     save_mode="reuse",
#     batch_size=5000,
#     verbose=True,
#     names=["no_pruning"]
# )

# del dense_e2e

################################
# BERT token pruned calculations
################################

dense_e2e_bert_pruned = pytcolbert.end_to_end(set(), prune_queries=False, prune_documents=False, remap_special_toks=args.remap_special_toks, remap_masks=args.remap_masks)
if args.remap_special_toks:
    name = "trec_remap_special_toks"
elif args.remap_masks:
    name = "trec_remap_masks"
else:
    print("Must specify remapping strategy.")
    exit(1)
print(f"Experiment: {name} to nearest query embedding.")
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
    names=[name]
)

del dense_e2e_bert_pruned

