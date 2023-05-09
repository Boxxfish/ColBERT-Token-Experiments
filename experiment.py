import numpy as np
import pyterrier as pt
# Initialize pyterrier
if not pt.started():
    pt.init()

from pyterrier_colbert.ranking import ColBERTFactory
from ir_measures import RR

#create a ColBERT ranking factory based on the pretrained checkpoint
pytcolbert = ColBERTFactory("http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip", "./msmarco_index", "msmarco", gpu=True)

# Download and initialize the msmarco dataset
msmarco_ds = pt.get_dataset("msmarco_passage")

# Get list of usable queries that have a corresponding relevant in our limited index
qids = list(msmarco_ds.get_qrels("dev").loc[msmarco_ds.get_qrels("dev")['docno'].astype(int) < 100000]['qid'])


# Run document pruning experiments from no pruning to 100% pruning at 5% increments
for p in np.arange(0.0, 1.01, 0.05):
    token_ids = pytcolbert.top_p_tokens(p)
    dense_e2e = pytcolbert.end_to_end(token_ids if p != 0 else None)

    result = pt.Experiment(
        [dense_e2e],
        msmarco_ds.get_topics("dev").loc[msmarco_ds.get_topics("dev")['qid'].str.contains('|'.join(qids), na=False)],
        msmarco_ds.get_qrels("dev"),
        eval_metrics=["map", RR@10],
        names=[f"ColBERT ({p=:0.2f})"]
    )

    print(result)

    # Must delete dense_e2e value to free GPU memory
    del dense_e2e


# Compare no pruning with p=0.5
dense_e2e_p0 = pytcolbert.end_to_end()

token_ids_p05 = pytcolbert.top_p_tokens(0.5)
dense_e2e_p05 = pytcolbert.end_to_end(token_ids_p05)

pt.Experiment(
    [dense_e2e_p0, dense_e2e_p05],
    msmarco_ds.get_topics("dev").loc[msmarco_ds.get_topics("dev")['qid'].str.contains('|'.join(qids), na=False)],
    msmarco_ds.get_qrels("dev"),
    eval_metrics=["map", RR@10],
    baseline=0,
    names=["p=0.0", "p=0.5"]
)


# Get list of BERT tokens 
ids = pytcolbert.unique_token_ids

bert_ids = []
for id, tok in zip(ids, pytcolbert.token_ids_to_strings(ids)):
    if tok.startswith('[') and tok.endswith(']'):
        print(tok, id)
        bert_ids += [id]

# Uncomment this line to only prune '[CLS]' and '[SEP]'
# bert_ids = [101, 102]


# Compare the effectiveness of the model before and after only BERT tokens are pruned
dense_e2e = pytcolbert.end_to_end()
dense_e2e_bert_pruned = pytcolbert.end_to_end(bert_ids)

pt.Experiment(
    [dense_e2e, dense_e2e_bert_pruned],
    msmarco_ds.get_topics("dev").loc[msmarco_ds.get_topics("dev")['qid'].str.contains('|'.join(qids), na=False)],
    msmarco_ds.get_qrels("dev"),
    eval_metrics=["map", RR@10],
    baseline=0,
    names=["No Pruning", "BERT Tokens Pruned"]
)