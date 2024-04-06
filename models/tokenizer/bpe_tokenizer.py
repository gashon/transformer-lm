import os
import regex as re
import collections
from models.tokenizer.vocab import Vocab

class BPETokenizer:
    def __init__(self, vocab_size: int, special_tokens: list[str] = []) -> None:
        """
        Initialize the BPETokenizer.

        Args:
            vocab_size: The desired size of the vocabulary.
            special_tokens: A list of special tokens to be added to the vocabulary.
        """
        self.vocab = Vocab(special_tokens=special_tokens)
        self.merges: list[tuple[bytes, bytes]] = []
        self.special_tokens = set(special_tokens)
        self.vocab_size = vocab_size

        # gpt-2 pre-tokenizer
        # @see https://github.com/openai/tiktoken/pull/234
        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", re.UNICODE)

    def from_file(self, input_path: str | os.PathLike) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Build the vocabulary and merges from an input file.

        Args:
            input_path: The path to the input file.

        Returns:
            A tuple containing:
                - The vocabulary mapping indices to tokens.
                - The list of merged byte pairs.
        """
        with open(input_path, 'r', encoding='utf-8') as file:
            text = file.read()

        presubwords = self._get_presubwords(text, self.PAT, self.special_tokens)
        subwords = self._encode_presubwords(presubwords)
        byte_pairs, token_indices = self._calculate_byte_pair_stats(subwords)

        while len(self.vocab) < self.vocab_size and byte_pairs:
            if byte_pairs[max(byte_pairs, key=lambda x: (byte_pairs[x], x))] < 2:
                break
            self._merge_best_pair(subwords, byte_pairs, token_indices, self.merges, self.vocab)

        return (self.vocab.get_idx_to_token(), self.merges)


    def encode(self, text: str) -> list[int]:
        """
        Encode the input text into a list of token indices.

        Args:
            text: The input text to be encoded.

        Returns:
            A list of token indices.
        """
        presubwords = self._get_presubwords(text, self.PAT, self.special_tokens)
        subwords = self._encode_presubwords(presubwords)
        token_indices = []
        for word in subwords:
            for i in range(len(word)):
                for j in range(len(self.merges)):
                    if word[i:i+2] == list(self.merges[j]):
                        word[i:i+2] = [self.merges[j][0] + self.merges[j][1]]
                        break
            token_indices.extend(self.vocab.get_token_to_idx()[token] for token in word)
        return token_indices

    def decode(self, token_indices: list[int]) -> str:
        """
        Decode a list of token indices into text.

        Args:
            token_indices: A list of token indices to be decoded.

        Returns:
            The decoded text.
        """
        tokens = [self.vocab.get_idx_to_token()[idx] for idx in token_indices]
        return ''.join(token.decode('utf-8') for token in tokens)

    @staticmethod
    def _get_presubwords(text: str, pat: re.Pattern, special_tokens: set[str]) -> list[str]:
        """
        Tokenize the input text into pre-subwords using the pre-tokenizer pattern.

        Args:
            text: The input text to be tokenized.
            pat: The pre-tokenizer pattern.
            special_tokens: A set of special tokens.

        Returns:
            A list of pre-subwords.
        """
        pre_subwords: list[str] = pat.findall(text)
        return [token for token in pre_subwords if token not in special_tokens]

    @staticmethod
    def _encode_presubwords(presubwords: list[str]) -> list[list[bytes]]:
        """
        Encode the pre-subwords into bytes.

        Args:
            presubwords: A list of pre-subwords to be encoded.

        Returns:
            A list of encoded subwords, where each subword is a list of bytes.
        """
        return [
            [bytes([b]) for b in token.encode("utf-8")]
            for token in presubwords
        ]

    @staticmethod
    def _calculate_byte_pair_stats(subwords: list[list[bytes]]) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], list[int]]]:
        """
        Calculate the frequency of byte pairs and the token indices.

        Args:
            subwords: A list of encoded subwords.

        Returns:
            A tuple containing:
                - byte_pairs: A dictionary mapping byte pairs to their frequency.
                - token_indices: A dictionary mapping byte pairs to the indices of tokens where they occur.
        """
        byte_pairs: dict[tuple[bytes, bytes], int] = collections.defaultdict(int)
        token_indices: dict[tuple[bytes, bytes], list[int]] = collections.defaultdict(list)
        for i, word in enumerate(subwords):
            for j in range(len(word) - 1):
                byte_pairs[(word[j], word[j+1])] += 1
                token_indices[(word[j], word[j+1])].append(i)
        return byte_pairs, token_indices

    @staticmethod
    def _merge_best_pair(subwords: list[list[bytes]], byte_pairs: dict[tuple[bytes, bytes], int], token_indices: dict[tuple[bytes, bytes], list[int]], merges: list[tuple[bytes, bytes]], vocab: Vocab) -> None:
        """
        Merge the best byte pair and update the subwords and vocabulary.

        Args:
            subwords: A list of encoded subwords.
            byte_pairs: A dictionary mapping byte pairs to their frequency.
            token_indices: A dictionary mapping byte pairs to the indices of tokens where they occur.
            merges: The list of merged byte pairs.
            vocab: The vocabulary to be updated.
        """
        best_pair = max(byte_pairs, key=lambda x: (byte_pairs[x], x))
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab.add_token(new_token)

        indices = token_indices[best_pair]
        for i in indices:
            for j, (left, right) in enumerate(zip(subwords[i][:-1], subwords[i][1:])):
                if left == best_pair[0] and right == best_pair[1]:
                    BPETokenizer._update_byte_pair_stats(subwords, i, j, new_token, byte_pairs, token_indices)
                    subwords[i][j:j+2] = [new_token]
                    break

        byte_pairs.pop(best_pair)
        token_indices.pop(best_pair)

    @staticmethod
    def _update_byte_pair_stats(subwords: list[list[bytes]], i: int, j: int, new_token: bytes, byte_pairs: dict[tuple[bytes, bytes], int], token_indices: dict[tuple[bytes, bytes], list[int]]) -> None:
        """
        Update the byte pair statistics after merging a pair.

        Args:
            subwords: A list of encoded subwords.
            i: The index of the current word.
            j: The index of the current byte pair within the word.
            new_token: The new token created by merging the byte pair.
            byte_pairs: A dictionary mapping byte pairs to their frequency.
            token_indices: A dictionary mapping byte pairs to the indices of tokens where they occur.
        """
        if j > 0:
            left_pair = (subwords[i][j-1], subwords[i][j])
            byte_pairs[left_pair] -= 1
            byte_pairs[(subwords[i][j-1], new_token)] += 1
            token_indices[(subwords[i][j-1], new_token)].append(i)
        if j < len(subwords[i]) - 2:
            right_pair = (subwords[i][j+1], subwords[i][j+2])
            byte_pairs[right_pair] -= 1
            byte_pairs[(new_token, subwords[i][j+2])] += 1
            token_indices[(new_token, subwords[i][j+2])].append(i)
