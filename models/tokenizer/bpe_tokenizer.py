import os
import regex as re
import collections
from models.tokenizer.vocab import Vocab

class BPETokenizer:
    def __init__(self, vocab_size: int, special_tokens: list[str] = []) -> None:
        self.vocab = Vocab(special_tokens=special_tokens) 
        self.merges: list[tuple[bytes, bytes]] = [] 
        self.special_tokens = set(special_tokens)
        self.vocab_size = vocab_size

        # gpt-2 pre-tokenizer 
        # @see https://github.com/openai/tiktoken/pull/234
        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", re.UNICODE)

    def get_presubwords(self, text: str) -> list[str]:
        pre_subwords: list[str] = self.PAT.findall(text)
        return [token for token in pre_subwords if token not in self.special_tokens]

    def from_file(self, input_path: str | os.PathLike) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        with open(input_path, 'r', encoding='utf-8') as file:
            text = file.read()

        presubwords = self.get_presubwords(text)

        subwords: list[list[bytes]] = [
            [bytes([b]) for b in token.encode("utf-8")]
            for token in presubwords
        ]

        token_indices: dict[tuple[bytes, bytes], list[int]] = collections.defaultdict(list)
        byte_pairs: dict[tuple[bytes, bytes], int] = collections.defaultdict(int)
        for i, word in enumerate(subwords):
            for j in range(len(word) - 1):
                byte_pairs[(word[j], word[j+1])] += 1
                token_indices[(word[j], word[j+1])].append(i)

        while len(self.vocab) < self.vocab_size and byte_pairs:
            # find the best pair
            best_pair = max(byte_pairs, key=lambda x: (byte_pairs[x], x))
            if byte_pairs[best_pair] < 2:
                break

            self.merges.append(best_pair)

            # add new token to vocab
            new_token = best_pair[0] + best_pair[1]
            self.vocab.add_token(new_token)

            indices = token_indices[best_pair]

            for i in indices:
                for j, (left, right) in enumerate(zip(subwords[i][:-1], subwords[i][1:])):
                    if left == best_pair[0] and right == best_pair[1]:
                        if j > 0:
                            left_pair = (subwords[i][j-1], left)
                            byte_pairs[left_pair] -= 1
                            byte_pairs[(subwords[i][j-1], new_token)] += 1
                            token_indices[(subwords[i][j-1], new_token)].append(i)
                        if j < len(subwords[i]) - 2:
                            right_pair = (right, subwords[i][j+2])
                            byte_pairs[right_pair] -= 1
                            byte_pairs[(new_token, subwords[i][j+2])] += 1
                            token_indices[(new_token, subwords[i][j+2])].append(i)
                        subwords[i][j:j+2] = [new_token]
                        break
            
            byte_pairs.pop(best_pair)
            token_indices.pop(best_pair)

        return (self.vocab.get_idx_to_token(), self.merges)
