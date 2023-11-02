"""
A little CLI demo that shows how query tokens are weighted with MASK and Q tokens.
"""
import pandas as pd
import torch
import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_colbert.ranking import ColBERTFactory
from transformers import AutoTokenizer

def main():
    pytcolbert = ColBERTFactory("http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip", "./msmarco_index", "msmarco", gpu=True)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    while True:
        query = input(">")
        Q, q_tok_ids, masks = pytcolbert.args.inference.queryFromText([query], with_ids=True)
        q_tok_ids = q_tok_ids[0].cpu()
        Q_f = Q[0:1, :, :]

        _Q, CLS, SEP, MASK = (1, 101, 102, 103)
        sep_index = torch.where(q_tok_ids.squeeze() == SEP)[0].item()
        remap_idxs = [1] + list(range(sep_index + 1, 32))
        
        remap_mask = torch.zeros(32, device=Q_f.device, dtype=torch.bool)
        for remap_idx in remap_idxs:
            remap_mask[remap_idx] = True
    
        weight_dict = {}
        for i in [0] + list(range(2, sep_index + 1)):
            weight_dict[i] = 1
        
        # Using discrete weights
        # dists = Q_f[0] @ Q_f[0].T # Shape: (32, 32)
        # dists = torch.masked_fill(dists, remap_mask, -float("inf"))
        # mapped_tok_idxs = torch.argmax(dists, 1).cpu()
        # for i in remap_idxs:
        #     weight_dict[mapped_tok_idxs[i].item()] += 1
        
        # Using interpolated weights
        # dists = Q_f[0] @ Q_f[0].T # Shape: (32, 32)
        # dists = (torch.masked_fill(dists, ~remap_mask, 0.0) + 1) / 2.0
        # dists_summed = dists.sum(1)
        # for i in [0] + list(range(2, sep_index + 1)):
        #     weight_dict[i] += dists_summed[i].item()

        # for i in [0] + list(range(2, sep_index + 1)):
        #     print(tokenizer.convert_ids_to_tokens(q_tok_ids[i].item()), ":", weight_dict[i])

        # Computing probability distributions
        dists = Q_f[0] @ Q_f[0].T # Shape: (32, 32)
        dists = torch.masked_fill(dists, remap_mask, -float("inf"))
        all_logits = []
        for target_mask in remap_idxs:
            logits = torch.softmax(dists[target_mask], 0)
            logits_per_token = []
            for i in [0] + list(range(2, sep_index + 1)):
                logits_per_token.append(logits.item())
                # print("MASK", target_mask, ",", tokenizer.convert_ids_to_tokens(q_tok_ids[i].item()), ":", logits[i])
            all_logits.append(logits_per_token)
        




if __name__ == "__main__":
    main()