import collections
from typing import Iterable, Iterator, List, Tuple, Dict
import os
import json
import regex as re

class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] = []):
        self.vocab = vocab
        self.vocab_inv = {v: k for k, v in self.vocab.items()}
        self.special_tokens = special_tokens or []
        self.merges = collections.defaultdict(dict)
        for i, (a, b) in enumerate(merges):
            self.merges[(a, b)] = i

        # gpt-2 pre-tokenizer
        # @see https://github.com/openai/tiktoken/pull/234
        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", re.UNICODE)

        # Add special tokens to vocabulary if not present
        for token in self.special_tokens:
            if token.encode('utf-8') not in self.vocab_inv:
                self.vocab[len(self.vocab)] = token.encode('utf-8')
                self.vocab_inv[token.encode('utf-8')] = len(self.vocab) - 1

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] = []):
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            merges = json.load(f)
        vocab = {int(k): bytes(v, 'utf-8') for k, v in vocab.items()}
        merges = [(bytes(a, 'utf-8'), bytes(b, 'utf-8')) for a, b in merges]
        return cls(vocab, merges, special_tokens)

    def pretokenize(self, text: str) -> List[str]:
        matches = [] 

        for match in self.PAT.finditer(text):
            match_str = match.group(0)
            if match_str in self.special_tokens:
                continue
            matches.append(match_str)

        return matches 

    def merge(self, tokens: List[bytes], pair: Tuple[bytes, bytes], replacement: bytes) -> List[bytes]:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if tokens[i] == pair[0] and i < len(tokens) - 1 and tokens[i + 1] == pair[1]:
                new_tokens.append(replacement)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        return new_tokens

    def encode(self, text: str) -> List[int]:
        # Pre-tokenize
        pre_token_count = self.pretokenize(text)
        print("Pre-token count: ", pre_token_count)
        
        ids = []
        for token in pre_token_count:
            raw_bytes = [bytes([b]) for b in token.encode('utf-8')]
            token_ids = []

            while len(raw_bytes) > 1:
                pairs = [pair for pair in zip(raw_bytes, raw_bytes[1:])]
                pair = min(pairs, key= lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                raw_bytes = self.merge(raw_bytes, pair, pair[0] + pair[1])
            for byte in raw_bytes:
                token_ids.append(self.vocab_inv[byte])
            ids.extend(token_ids)

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: List[int]) -> str:
        raw_bytes = b"".join([self.vocab[i] for i in ids])
        return raw_bytes.decode('utf-8', errors='replace')

    def _get_stats(self, tokens: List[bytes]) -> Dict[Tuple[bytes, bytes], int]:
        counts = collections.defaultdict(int)
        for pair in zip(tokens, tokens[1:]):
            counts[pair] += 1
        return counts

