import argparse

import numpy as np
import torch

from models.tokenizer.tokenizer import Tokenizer

fname = {
    "tiny/train": "TinyStoriesV2-GPT4-train.txt",
    "tiny/valid": "TinyStoriesV2-GPT4-valid.txt",
    "owt/train": "owt_train.txt",
    "owt/valid": "owt_valid.txt",
}


def main(args):
    with open(f"/data/{fname[args.dataset+'/'+args.split]}", "r") as f:
        text = f.read()

    tokenizer = Tokenizer.from_files(
        vocab_filepath=f"data/tokenizer/{args.dataset}-vocab.pkl",
        merges_filepath=f"data/tokenizer/{args.dataset}-merges.pkl",
        special_tokens=["<|endoftext|>"],
    )

    tokens = tokenizer.encode(text)
    pt = np.array(tokens, dtype=np.uint16)

    torch.save(
        pt, f"data/tokenizer/{args.dataset}-tokens-{args.split}.pt", pickle_protocol=4
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--split", type=str)

    args = parser.parse_args()
    main(args)
