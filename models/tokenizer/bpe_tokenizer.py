import collections
import json
import os
import regex as re
from typing import List, Tuple, Dict

from memory_profiler import profile
from models.tokenizer.vocab import Vocab
import psutil


class BPETokenizer:
    def __init__(self, vocab_size: int, special_tokens: List[str] = []) -> None:
        """
        Initialize the BPETokenizer.

        Args:
            vocab_size: The desired size of the vocabulary.
            special_tokens: A list of special tokens to be added to the vocabulary.
        """
        self.vocab = Vocab(special_tokens=special_tokens)
        self.merges: List[Tuple[bytes, bytes]] = []
        self.special_tokens = set(special_tokens)
        self.vocab_size = vocab_size

        # gpt-2 pre-tokenizer
        # @see https://github.com/openai/tiktoken/pull/234
        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", re.UNICODE)

    def from_file(self, input_path: str | os.PathLike) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        """
        Build the vocabulary and merges from an input file.

        Args:
            input_path: The path to the input file.

        Returns:
            A tuple containing:
                - The vocabulary mapping indices to tokens.
                - The list of merged byte pairs.
        """
        print("getting presubwords")

        pretoken_counts = self.get_presubwords(input_path)
        unique_pretokens = list(pretoken_counts.keys())
        subwords = self._encode_presubwords(unique_pretokens)
        byte_pairs, token_indices = self._calculate_byte_pair_stats(subwords, pretoken_counts)

        while len(self.vocab) < self.vocab_size and byte_pairs:
            if self._get_max_byte_pair_frequency(byte_pairs) < 2:
                break
            self._merge_best_pair(subwords, byte_pairs, token_indices, self.merges, self.vocab)

        return self.vocab.get_idx_to_token(), self.merges

    def ends_with_special_token(self, line: str) -> bool:
        return any(line.strip().endswith(token) for token in self.special_tokens)

    def get_presubwords(self, input_path: str | os.PathLike) -> Dict[str, int]:
        match_count: Dict[str, int] = {}

        with open(input_path, 'r') as file:
            chunk = []
            for line in file:
                if self.ends_with_special_token(line):
                    self._update_match_count(chunk, match_count)
                    chunk = []
                else:
                    chunk.append(line)

        if chunk:
            self._update_match_count(chunk, match_count)

        return match_count

    def _update_match_count(self, chunk: List[str], match_count: Dict[str, int]) -> None:
        chunk_str = ''.join(chunk)
        matches = self.PAT.findall(chunk_str)
        for match in matches:
            if match not in self.special_tokens:
                match_count[match] = match_count.get(match, 0) + 1

    @staticmethod
    def _encode_presubwords(presubwords: List[str]) -> List[List[bytes]]:
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
    def _calculate_byte_pair_stats(
        subwords: List[List[bytes]],
        pretoken_counts: Dict[str, int]
    ) -> Tuple[Dict[Tuple[bytes, bytes], int], Dict[Tuple[bytes, bytes], Dict[int, Dict[str, List[int]]]]]:
        """
        Calculate the frequency of byte pairs and the token indices.

        Args:
            subwords: A list of encoded subwords.
            pretoken_counts: A dictionary mapping pre-subwords to their counts.

        Returns:
            A tuple containing:
                - byte_pairs: A dictionary mapping byte pairs to their frequency.
                - token_indices: A dictionary mapping byte pairs to the indices of tokens (key) and number of occurrences (value). 
        """

        byte_pairs: Dict[Tuple[bytes, bytes], int] = collections.defaultdict(int)
        token_indices: Dict[Tuple[bytes, bytes], Dict[int, Dict[str, List[int]]]] = {}

        for i, word in enumerate(subwords):
            count = pretoken_counts[bytes(b''.join(word)).decode('utf-8')]

            for j in range(len(word) - 1):
                byte_pair = (word[j], word[j+1])
                byte_pairs[byte_pair] = byte_pairs.get(byte_pair, 0) + count
                if byte_pair not in token_indices:
                    token_indices[byte_pair] = {}
                if i not in token_indices[byte_pair]:
                    token_indices[byte_pair][i] = {"byte_idx": []}
                token_indices[byte_pair][i]["count"] = count
                token_indices[byte_pair][i]["byte_idx"].append(j)

        return byte_pairs, token_indices

    @staticmethod
    def _get_max_byte_pair_frequency(byte_pairs: Dict[Tuple[bytes, bytes], int]) -> int:
        """
        Get the maximum frequency of byte pairs.

        Args:
            byte_pairs: A dictionary mapping byte pairs to their frequency.

        Returns:
            The maximum frequency of byte pairs.
        """
        return byte_pairs[max(byte_pairs, key=lambda x: (byte_pairs[x], x))]

    @staticmethod
    def _merge_best_pair(
        subwords: List[List[bytes]],
        byte_pairs: Dict[Tuple[bytes, bytes], int],
        token_indices: Dict[Tuple[bytes, bytes], Dict[int, Dict[str, List[int]]]],
        merges: List[Tuple[bytes, bytes]],
        vocab: Vocab
    ) -> None:
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

        tkn_idxs = token_indices[best_pair].keys()
        for tkn_idx in tkn_idxs:
            byte_idxs = token_indices[best_pair][tkn_idx]["byte_idx"]

            for byte_idx in byte_idxs:
                assert subwords[tkn_idx][byte_idx] == best_pair[0]
                assert subwords[tkn_idx][byte_idx + 1] == best_pair[1]

                count = token_indices[best_pair][tkn_idx]["count"]

                BPETokenizer._update_byte_pair_stats(subwords, tkn_idx, byte_idx, new_token, byte_pairs, token_indices, count)

            merged_idxs = byte_idxs
            for merged_idx in merged_idxs:
                BPETokenizer._update_token_indices_after_merge(subwords, tkn_idx, merged_idx, token_indices)

            BPETokenizer._merge_subwords(subwords, tkn_idx, byte_idxs, new_token)
            merged_idxs = [idx-pos for pos, idx in enumerate(byte_idxs)]

            for merged_idx in merged_idxs:
                BPETokenizer._create_new_token_indices(subwords, tkn_idx, merged_idx, token_indices, best_pair)

            BPETokenizer._reset_token_indices(subwords, tkn_idx, token_indices)

        byte_pairs.pop(best_pair)
        token_indices.pop(best_pair)

    @staticmethod
    def _update_token_indices_after_merge(
        subwords: List[List[bytes]],
        tkn_idx: int,
        merged_idx: int,
        token_indices: Dict[Tuple[bytes, bytes], Dict[int, Dict[str, List[int]]]]
    ) -> None:
        """
        Update token indices after merging a pair.

        Args:
            subwords: A list of encoded subwords.
            tkn_idx: The index of the current word.
            merged_idx: The index of the merged byte pair within the word.
            token_indices: A dictionary mapping byte pairs to the indices of tokens where they occur.
        """
        if merged_idx > 0:
            pair = (subwords[tkn_idx][merged_idx - 1], subwords[tkn_idx][merged_idx])
            if merged_idx - 1 in token_indices[pair][tkn_idx]["byte_idx"]:
                token_indices[pair][tkn_idx]["byte_idx"].remove(merged_idx - 1)
        if merged_idx < len(subwords[tkn_idx]) - 2:
            pair = (subwords[tkn_idx][merged_idx+1], subwords[tkn_idx][merged_idx + 2])
            if merged_idx + 1 in token_indices[pair][tkn_idx]["byte_idx"]:
                token_indices[pair][tkn_idx]["byte_idx"].remove(merged_idx + 1)

    @staticmethod
    def _merge_subwords(
        subwords: List[List[bytes]],
        tkn_idx: int,
        byte_idxs: List[int],
        new_token: bytes
    ) -> None:
        """
        Merge subwords at the given byte indices.

        Args:
            subwords: A list of encoded subwords.
            tkn_idx: The index of the current word.
            byte_idxs: The indices of the byte pairs to be merged.
            new_token: The new token created by merging the byte pairs.
        """
        for byte_idx in byte_idxs[::-1]:
            subwords[tkn_idx][byte_idx] = new_token
            subwords[tkn_idx].pop(byte_idx + 1)

    @staticmethod
    def _create_new_token_indices(
        subwords: List[List[bytes]],
        tkn_idx: int,
        merged_idx: int,
        token_indices: Dict[Tuple[bytes, bytes], Dict[int, Dict[str, List[int]]]],
        best_pair: Tuple[bytes, bytes]
    ) -> None:
        """
        Create new token indices for merged pairs.

        Args:
            subwords: A list of encoded subwords.
            tkn_idx: The index of the current word.
            merged_idx: The index of the merged byte pair within the word.
            token_indices: A dictionary mapping byte pairs to the indices of tokens where they occur.
            best_pair: The best byte pair to be merged.
        """
        count = token_indices[best_pair][tkn_idx]["count"]
        if merged_idx > 0:
            left_pair = (subwords[tkn_idx][merged_idx - 1], subwords[tkn_idx][merged_idx])
            if left_pair not in token_indices:
                token_indices[left_pair] = {}
            token_indices[left_pair][tkn_idx] = {"count": count, "byte_idx": []}
            token_indices[left_pair][tkn_idx]["byte_idx"].append(merged_idx - 1)

        if merged_idx < len(subwords[tkn_idx]) - 1:
            right_pair = (subwords[tkn_idx][merged_idx], subwords[tkn_idx][merged_idx + 1])
            if right_pair not in token_indices:
                token_indices[right_pair] = {}
            token_indices[right_pair][tkn_idx] = {"count": count, "byte_idx": []}
            token_indices[right_pair][tkn_idx]["byte_idx"].append(merged_idx)

    @staticmethod
    def _reset_token_indices(
        subwords: List[List[bytes]],
        tkn_idx: int,
        token_indices: Dict[Tuple[bytes, bytes], Dict[int, Dict[str, List[int]]]]
    ) -> None:
        """
        Reset token indices after merging pairs.

        Args:
            subwords: A list of encoded subwords.
            tkn_idx: The index of the current word.
            token_indices: A dictionary mapping byte pairs to the indices of tokens where they occur.
        """
        for byte_idx in range(len(subwords[tkn_idx]) - 1):
            pair = (subwords[tkn_idx][byte_idx], subwords[tkn_idx][byte_idx + 1])
            token_indices[pair][tkn_idx]["byte_idx"] = []

        for byte_idx in range(len(subwords[tkn_idx]) - 1):
            pair = (subwords[tkn_idx][byte_idx], subwords[tkn_idx][byte_idx + 1])
            token_indices[pair][tkn_idx]["byte_idx"].append(byte_idx)

    @staticmethod
    def _update_byte_pair_stats(
        subwords: List[List[bytes]],
        i: int,
        j: int,
        new_token: bytes,
        byte_pairs: Dict[Tuple[bytes, bytes], int],
        token_indices: Dict[Tuple[bytes, bytes], Dict[int, Dict[str, List[int]]]],
        freq: int
    ) -> None:
        """
        Update the byte pair statistics after merging a pair.

        Args:
            subwords: A list of encoded subwords.
            i: The index of the current word.
            j: The index of the current byte pair within the word.
            new_token: The new token created by merging the byte pair.
            byte_pairs: A dictionary mapping byte pairs to their frequency.
            token_indices: A dictionary mapping byte pairs to the indices of tokens where they occur.
            freq: The frequency of the merged pair.
        """
        if j > 0:
            left_pair = (subwords[i][j-1], subwords[i][j])
            byte_pairs[left_pair] -= freq
            byte_pairs[(subwords[i][j-1], new_token)] += freq

        if j < len(subwords[i]) - 2:
            right_pair = (subwords[i][j+1], subwords[i][j+2])
            byte_pairs[right_pair] -= freq
            byte_pairs[(new_token, subwords[i][j+2])] += freq

    def save_to_file(self, vocab_file: str, merges_file: str) -> None:
        """
        Save the vocabulary and merges to files for further inspection.

        Args:
            vocab_file: The path to the file where the vocabulary will be saved.
            merges_file: The path to the file where the merges will be saved.
        """
        self._save_vocab(vocab_file)
        self._save_merges(merges_file)

    def _save_vocab(self, vocab_file: str) -> None:
        """
        Save the vocabulary to a file.

        Args:
            vocab_file: The path to the file where the vocabulary will be saved.
        """
        with open(vocab_file, 'w', encoding='utf-8') as f:
            vocab_data = {
                'vocab_size': self.vocab_size,
                'special_tokens': list(self.special_tokens),
                'vocab': {}
            }
            for idx, token in self.vocab.get_idx_to_token().items():
                decoded_token = self._decode_token(token)
                vocab_data['vocab'][idx] = decoded_token
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    def _save_merges(self, merges_file: str) -> None:
        """
        Save the merges to a file.

        Args:
            merges_file: The path to the file where the merges will be saved.
        """
        with open(merges_file, 'w', encoding='utf-8') as f:
            merges_data = []
            for merge in self.merges:
                decoded_merge = self._decode_merge(merge)
                merges_data.append(decoded_merge)
            json.dump(merges_data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _decode_token(token: bytes) -> str:
        """
        Decode a token from bytes to string.

        Args:
            token: The token to be decoded.

        Returns:
            The decoded token as a string.
        """
        try:
            decoded_token = token.decode('utf-8')
        except UnicodeDecodeError:
            decoded_token = f'<byte_{token.hex()}>'
        return decoded_token

    @staticmethod
    def _decode_merge(merge: Tuple[bytes, bytes]) -> Tuple[str, str]:
        """
        Decode a merge from bytes to string.

        Args:
            merge: The merge to be decoded.

        Returns:
            The decoded merge as a tuple of strings.
        """
        try:
            decoded_merge = (merge[0].decode('utf-8'), merge[1].decode('utf-8'))
        except UnicodeDecodeError:
            decoded_merge = (f'<byte_{merge[0].hex()}>', f'<byte_{merge[1].hex()}>')
        return decoded_merge
