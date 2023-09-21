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
parser.add_argument("--no-pruning", action="store_true")
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

# Token IDs.
Q = "[unused0]"
prune_tokens = [Q, "[SEP]", "[MASK]", "[CLS]"]
prune_set = set(tokenizer.convert_tokens_to_ids(prune_tokens))
if args.no_pruning:
    pass
else:
    tokenized_ids: list = list(set(sum(tokenizer(stopwords.words("english"), add_special_tokens=False).input_ids, [])))
    prune_set = prune_set.union(set(tokenized_ids))
prune_tokens_str = tokenizer.convert_ids_to_tokens(list(prune_set))

dense_e2e_bert_pruned = pytcolbert.end_to_end(prune_set, prune_queries=True, prune_documents=False)
print(f"Experiment: Tokens to prune: {prune_tokens_str}")
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
    names=["trec_" + ("pruned_stopwords" if not args.no_pruning else "no_pruning")]
)

del dense_e2e_bert_pruned

