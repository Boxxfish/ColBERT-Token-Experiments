"""
Utilities for modifying the query before contextualization and after.
"""
from typing import *
import torch
from torch import Tensor
from colbert.modeling.inference import ModelInference
import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_colbert.ranking import ColBERTFactory

# Special token IDs
PAD, Q, CLS, SEP, MASK = (0, 1, 101, 102, 103)

def replace_q(replace_q_tok: int):
    def _replace_q(qtoks: torch.Tensor) -> torch.Tensor:
        qtoks[1] = replace_q_tok
        return qtoks
    return _replace_q

def remap_special_toks_or_remap_masks(remap_special_toks: bool, remap_masks: bool, remap_masks_to_terms: bool):
    assert(sum([remap_special_toks, remap_masks, remap_masks_to_terms]) <= 1)
    def _remap_special_toks_or_remap_masks(qtoks: torch.Tensor, qembs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # If remapping all special tokens, we also include SEP, Q, and CLS. Otherwise, we only remap MASKs.
        sep_index = torch.where(qtoks.squeeze() == SEP)[0].item()
        if remap_special_toks:
            remap_idxs = [0, 1] + list(range(sep_index, 32))
        elif remap_masks or remap_masks_to_terms:
            remap_idxs = list(range(sep_index + 1, 32))
        
        remap_mask = torch.zeros(32, device=qembs.device, dtype=torch.bool)
        for remap_idx in remap_idxs:
            remap_mask[remap_idx] = True

        # If we're remapping masks to only query terms, set the remap mask to True on the special tokens
        if remap_masks_to_terms:
            remap_mask[0] = True
            remap_mask[1] = True
            remap_mask[sep_index] = True


        dists = qembs @ qembs.T # Shape: (32, 32)
        dists = torch.masked_fill(dists, remap_mask, -float("inf"))
        mapped_tok_idxs = torch.argmax(dists, 1)
        for i in remap_idxs:
            qembs[i] = qembs[mapped_tok_idxs[i]]
            qtoks[i] = qtoks[mapped_tok_idxs[i]]
        return (qtoks, qembs)
    return _remap_special_toks_or_remap_masks

def keep_tok(tok: int):
    def _keep_tok(qtoks: torch.Tensor, qembs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tok_to_keep = torch.where(qtoks.squeeze() == tok)[0].item()
        query_token_mask = [False] * 32
        query_token_mask[tok_to_keep] = True
        query_token_mask = torch.tensor(query_token_mask, device=qembs.device)
        qtoks = qtoks[query_token_mask, :]
        return (qtoks, qembs)
    return _keep_tok

def keep_pos(pos: int):
    def _keep_pos(qtoks: torch.Tensor, qembs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query_token_mask = [False] * 32
        query_token_mask[pos] = True
        query_token_mask = torch.tensor(query_token_mask, device=qembs.device)
        qtoks = qtoks[query_token_mask, :]
        return (qtoks, qembs)
    return _keep_pos

# Queries used with `test_query_mod` if no test queries are passed.
DEFAULT_QUERIES = [
    "cost of endless pools swim spa",
    "how long is a college hockey game",
    "treasury officer hong kong average salary",
]

class TestResult:
    query: str
    # Query tokens before `mod_qtoks`.
    orig_qtoks: list[int]
    # Query tokens after `mod_qtoks` but before contextualization.
    qtoks_before_ctx: list[int]
    # Query embeddings before `mod_qembs`.
    qembs: Tensor
    # Query embeddings after `mod_qembs`.
    qembs_after_mod_qembs: Tensor
    # Query tokens after `mod_qembs`.
    qtoks_after_mod_qembs: list[int]

def get_model() -> ModelInference:
    pytcolbert = ColBERTFactory("http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip", "./msmarco_index", "msmarco", gpu=True)
    return pytcolbert.args.inference

def test_query_mod(
        mod_qtoks: Optional[Callable[[Tensor], Tensor]] = None,
        mod_qembs: Optional[Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]] = None,
        test_queries: Optional[list[str]] = None,
        model: Optional[ModelInference] = None
    ) -> list[TestResult]:
    """
    Tests query modifications.
    """
    if test_queries is None:
        test_queries = DEFAULT_QUERIES
    
    if model is None:
        model = get_model()

    results = []
    for query in test_queries:
        result = TestResult()
        result.query = query
        with torch.no_grad():
            batches = model.query_tokenizer.tensorize([query], bsize=1)
            result.orig_qtoks = list(batches[0][0])
            if mod_qtoks:
                for (input_ids, _) in batches:
                    input_ids[0] = mod_qtoks(input_ids[0])
            batchesEmbs = [model.query(input_ids, attention_mask, to_cpu=False) for input_ids, attention_mask in batches]
            Q, q_tok_ids, _ = (torch.cat(batchesEmbs), torch.cat([ids for ids, _ in batches]), torch.cat([masks for _, masks in batches]))
        
        q_tok_ids = q_tok_ids[0].cpu()
        Q_f = Q[0:1, :, :]

        result.qtoks_before_ctx = list(q_tok_ids)
        result.qembs = Q_f[0].clone()

        masks = [False] * 32
        if mod_qembs:
            q_tok_ids, Q_f[0] = mod_qembs(q_tok_ids, Q_f[0])
        result.masks = masks
        result.qembs_after_mod_qembs = Q_f[0].clone()
        result.qtoks_after_mod_qembs = q_tok_ids

        results.append(result)
    return results

def set_retreive_change(
        query: str,
        factory: ColBERTFactory,
        mod_qtoks: Optional[Callable[[Tensor], Tensor]] = None,
        mod_qembs: Optional[Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]] = None,
    ) -> tuple[set[int], set[int]]:
    """
    Runs a query before and after modifications are made.
    """
    before_retriever = factory.set_retrieve(mod_qtoks=mod_qtoks, mod_qembs=mod_qembs)
    after_retriever = factory.set_retrieve(mod_qtoks=mod_qtoks, mod_qembs=mod_qembs)
    docids1 = set(before_retriever.search(query).docids)
    docids2 = set(after_retriever.search(query).docids)
    return (docids1, docids2)
