# Initialize pyterrier
import pandas as pd
import pyterrier as pt

from mod_utils import load_colbert, remap_special_toks_or_remap_masks
if not pt.started():
    pt.init()
from pyterrier_colbert.ranking import ColBERTFactory
from metrics import eval_metrics
from trec_utils import process_ds

# Argparse stuff
from argparse import ArgumentParser
parser = ArgumentParser()
# parser.add_argument("--remap-special-toks", action="store_true")
# parser.add_argument("--remap-masks", action="store_true")
parser.add_argument("--v2")
args = parser.parse_args()
suffix = "_v2" if args.v2 else ""

# Tokenization stuff
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

pytcolbert = load_colbert("trec", args.v2)
topic, qrels = process_ds()

print("No Pruning Experiment...")
e2e = pytcolbert.end_to_end(
    set(),
    prune_queries=False,
    prune_documents=False,
)
pt.Experiment(
    [e2e],
    topic,
    qrels,
    eval_metrics=eval_metrics,
    save_dir="results",
    save_mode="reuse",
    batch_size=10000,
    verbose=True,
    names=[f"trec_baseline{suffix}"]
)

del e2e

print("Remap MASKs to terms...")
e2e = pytcolbert.end_to_end(
    set(),
    prune_queries=False,
    prune_documents=False,
    mod_qembs=remap_special_toks_or_remap_masks(False, False, True)
)

pt.Experiment(
        [e2e],
        topic,
        qrels,
        filter_by_qrels=True,
        eval_metrics=eval_metrics,
        save_dir="results",
        save_mode="reuse",
        batch_size=10000,
        verbose=True,
        names=[f"trec_remap_masks_to_terms{suffix}"]
)

del e2e

for op in [True, False]:
    remap_special_toks = False
    remap_masks = False
    if op:
        remap_special_toks = True
    else:
        remap_masks = True
    e2e = pytcolbert.end_to_end(
        set(),
        prune_queries=False,
        prune_documents=False,
        mod_qembs=remap_special_toks_or_remap_masks(remap_special_toks, remap_masks)
    )
    if remap_special_toks:
        name = f"trec_remap_special_toks{suffix}"
    elif remap_masks:
        name = f"trec_remap_masks{suffix}"
    else:
        print("Must specify remapping strategy.")
        exit(1)
    print(f"Experiment: {name} to nearest query embedding.")
    pt.Experiment(
        [e2e],
        topic,
        qrels,
        filter_by_qrels=True,
        eval_metrics=eval_metrics,
        save_dir="results",
        save_mode="reuse",
        batch_size=10000,
        verbose=True,
        names=[name]
    )

    del e2e

