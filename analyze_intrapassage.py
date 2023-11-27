import random
import torch
from transformers import AutoTokenizer
from trec_utils import process_ds
import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_colbert.ranking import ColBERTFactory
import pandas as pd
from tqdm import tqdm
from mod_utils import SEP

PERIOD = 1012

def compute_maxsim(qembs: torch.Tensor, dembs: torch.Tensor) -> float:
    interaction = qembs @ dembs.T # Shape: (num_qembs, num_dembs)
    score = (
        interaction
            .amax(1, keepdim=False) # Shape: (num_qembs)
            .sum(0) #Shape: (1)
    )
    return score.item()

def query_correct(
        qid: int,
        topic: pd.DataFrame,
        qrels: pd.DataFrame,
        pytcolbert: ColBERTFactory,
        tokenizer: AutoTokenizer,
        strategy: str, # Either "all", "cls", or "sep".
        only_periods: bool,
    ):
    # Set up good and bad passages.
    num_bad = 1
    good = qrels.loc[(qrels.qid == str(qid)) & (qrels.label >= 2)].docno.tolist()[0]
    bads = qrels.loc[(qrels.qid == str(qid)) & (qrels.label == 0)].docno.tolist()[:num_bad]
    doc_list = list(enumerate([good] + bads))
    random.shuffle(doc_list)
    indices, doc_list = zip(*doc_list)

    # Concatenate document tokens together, keeping track of which tokens
    # belong to which passage.
    final_toks = []
    spans_good = [] # True if span belongs to good passage.
    span_offsets = []
    offset = 2
    for idx, docno in zip(indices, doc_list):
        docno = int(docno)
        toks = pytcolbert.nn_term().get_tokens_for_doc(docno)[2:-1]
        spans_good.append(idx == 0)
        span_offsets.append(offset)
        offset += len(toks)
        final_toks += toks
    span_offsets.append(offset)
    final_toks = final_toks[:179]
    
    # Compute embeddings for query and document.
    doc = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(final_toks))
    batches = pytcolbert.args.inference.doc_tokenizer.tensorize([doc], bsize=1)
    dembs = pytcolbert.args.inference.doc(batches[0][0][0], batches[0][0][1]).squeeze(0)
    qembs, qids, _ = pytcolbert.args.inference.queryFromText([topic.loc[topic.qid == str(qid)]["query"].item()], bsize=1, with_ids=True)
    qembs = qembs.squeeze(0)
    span_embs = []
    for i in range(1 + num_bad):
        start = span_offsets[i]
        end = span_offsets[i + 1]
        span = dembs[start:end]
        span_toks = doc[start:end]
        if only_periods:
            periods = [j for j, tok in enumerate(span_toks) if tok == PERIOD]
            print(span_toks)
            print(periods)
            span = span[periods]
            print(span)
        span_embs.append(span)
    qids = qids[0].squeeze(0)
    sep_idx = torch.where(qids == SEP)[0].item()
    assert(strategy in ["all", "cls", "sep"])
    if strategy == "cls":
        qembs = qembs[:1]
    if strategy == "sep":
        qembs = qembs[sep_idx:sep_idx + 1]

    # Score each intra-document span.
    best_score = -1
    best_score_good = False
    for i, span in enumerate(span_embs):
        span = span_embs[i]
        score = compute_maxsim(qembs, span)
        if score >= best_score:
            best_score = score
            best_score_good = spans_good[i]
    return best_score_good

def main():
    pytcolbert = ColBERTFactory("http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip", "./trec_index", "trec", gpu=False)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    topic, qrels = process_ds()

    for only_periods in [False, True]:
        for strategy in ["all", "cls", "sep"]:
            total = 0
            num_correct = 0
            for qid in tqdm(qrels.qid.unique()):
                qid = int(qid)
                random.seed(qid) # Ensure seed is the same across the QID.
                got_correct = query_correct(qid, topic, qrels, pytcolbert, tokenizer, strategy, only_periods)
                if got_correct:
                    num_correct += 1
                total += 1
            print(f"Only periods: {only_periods}, percent correct ({strategy}): {num_correct / total}")

if __name__ == "__main__":
    main()