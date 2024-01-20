"""
A utility script to convert a ColBERTv2 checkpoint into a ColBERTv1 checkpoint.
"""
from argparse import ArgumentParser
import torch
from pathlib import Path

def main():
    parser = ArgumentParser()
    parser.add_argument("--v2-dir")
    parser.add_argument("--out-file")
    args = parser.parse_args()

    v2_dir = Path(args.v2_dir)
    state_dict = torch.load(v2_dir / "pytorch_model.bin")

    chkpt = {
        "model_state_dict": state_dict,
        "epoch": 0,
        "batch": 0,
    }
    torch.save(chkpt, args.out_file)

if __name__ == "__main__":
    main()