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

    def pretokenize(self, text: str) -> Dict[str, int]:
        match_count = {} 

        for match in self.PAT.finditer(text):
            match_str = match.group(0)
            if match_str in self.special_tokens:
                continue
            match_count[match_str] = match_count.get(match_str, 0) + 1

        return match_count

    def merge(self, arr: List[bytes]):
        pairs = [pair for pair in zip(arr, arr[1:])] 
        while len(arr) >= 2:
            to_merge_pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if to_merge_pair not in self.merges:
                break
            for i in range(len(arr) - 1):
                if arr[i:i+2] == list(to_merge_pair):
                    arr[i:i+2] = [to_merge_pair[0] + to_merge_pair[1]]
                    i -= 1
                    
            pairs = [pair for pair in zip(arr, arr[1:])]

    def encode(self, text: str) -> List[int]:
        # Pre-tokenize
        pre_token_count = self.pretokenize(text)
        encoded_tokens = []

        for token in pre_token_count.keys():

            byte_arr = [bytes(char, 'utf-8') for char in token]
            # convert to byte_array
            subwords = []
            for byte_str in byte_arr:
                for byte in byte_str:
                    subwords.append(bytes([byte]))

            self.merge(subwords)
            encoded = []
            for byte in subwords:
                if byte in self.vocab_inv:
                    encoded.append(self.vocab_inv[byte])
                else:
                    # add to vocab and vocab_inv
                    self.vocab[len(self.vocab)] = byte 
                    self.vocab_inv[byte] = len(self.vocab) - 1
                    encoded.append(self.vocab_inv[byte])
                    # encoded.append(self.vocab_inv['<unk>'.encode('utf-8')])

            encoded_freq = pre_token_count[token]
            encoded_tokens.extend(encoded * encoded_freq)

        return encoded_tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: List[int]) -> str:
        if len(ids) == 0:
            return ''
        byte_agr = self.vocab[ids[0]]
        for id in ids[1:]:
            byte_agr = byte_agr + self.vocab[id]
        return byte_agr.decode('utf-8')
