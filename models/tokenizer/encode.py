import argparse

import numpy as np
import torch

from models.tokenizer.tokenizer import Tokenizer

fname = {
    "tiny/train": "TinyStoriesV2-GPT4-train.txt",
    "tiny/valid": "TinyStoriesV2-GPT4-valid.txt",
    "owt/train": "owt_train.txt",
    "owt/valid": "owt_valid.txt",
    "corpus/train": "corpus.en",
    "corpus/valid": "corpus.en",
}


def main(dataset: str, split: str):
    if dataset == "corpus":
        with open("tests/fixtures/corpus.en", "r") as f:
            text = f.read()
    else:
        with open(f"/data/{fname[dataset+'/'+split]}", "r") as f:
            text = f.read()

    tokenizer = Tokenizer.from_files(
        vocab_filepath=f"data/tokenizer/{dataset}-vocab.pkl",
        merges_filepath=f"data/tokenizer/{dataset}-merges.pkl",
        special_tokens=["<|endoftext|>"],
    )

    tokens = tokenizer.encode(text)
    pt = np.array(tokens, dtype=np.uint16)

    torch.save(pt, f"data/tokenizer/{dataset}-tokens-{split}.pt", pickle_protocol=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--split", type=str)

    args = parser.parse_args()
    main(args.dataset, args.split)
