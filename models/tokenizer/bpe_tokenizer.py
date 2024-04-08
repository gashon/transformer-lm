import os
import json
import regex as re
import collections
import psutil
from memory_profiler import profile
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
        print("getting presubwords")

        pretoken_counts = self.get_presubwords(input_path)
        unique_pretokens = list(pretoken_counts.keys())
        subwords = self._encode_presubwords(unique_pretokens)
        byte_pairs, token_indices = self._calculate_byte_pair_stats(subwords, pretoken_counts)

        while len(self.vocab) < self.vocab_size and byte_pairs:
            if byte_pairs[max(byte_pairs, key=lambda x: (byte_pairs[x], x))] < 2:
                break
            self._merge_best_pair(subwords, byte_pairs, token_indices, self.merges, self.vocab)

        return (self.vocab.get_idx_to_token(), self.merges)

    def ends_with_special_token(self, line:str):
        return any(line.strip().endswith(token) for token in self.special_tokens)

    def get_presubwords(self, input_path:str | os.PathLike) -> dict[str, int]:
        match_count = {}

        with open(input_path, 'r') as file:
            chunk = []
            for line in file:
                # Check if the current line ends with a special token
                if self.ends_with_special_token(line):
                    # Join the chunk into a single string for pattern matching
                    chunk_str = ''.join(chunk)
                    # Perform regex pattern matching on the chunk
                    matches = self.PAT.findall(chunk_str)
                    # Update the match count dictionary
                    for match in matches:
                        if match not in self.special_tokens:
                            match_count[match] = match_count.get(match, 0) + 1
                    # Reset the chunk for the next set of lines
                    chunk = []
                else:
                    # If not a special token, add the line to the current chunk
                    chunk.append(line)

        if chunk:
            chunk_str = ''.join(chunk)
            matches = self.PAT.findall(chunk_str)
            for match in matches:
                if match not in self.special_tokens:
                    match_count[match] = match_count.get(match, 0) + 1

        return match_count

    # def get_presubwords(self, input_path: str | os.PathLike, chunk_size: int = 1048576) -> dict[str, int]:
    #     print(f"Before processing - Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024} MB")
    #     pretoken_counts: dict[str, int] = {}
    #
    #     with open(input_path, 'r', encoding='utf-8') as file:
    #         chunk = ''
    #         overlap = ''
    #         while True:
    #             chunk += file.read(chunk_size)
    #             if not chunk:
    #                 break
    #
    #             # Find matches in the current chunk, including the overlap from the previous chunk
    #             pre_subwords: list[str] = self.PAT.findall(overlap + chunk)
    #
    #             for pretoken in pre_subwords:
    #                 if pretoken not in self.special_tokens:
    #                     pretoken_counts[pretoken] = pretoken_counts.get(pretoken, 0) + 1
    #
    #             # Update the overlap to include the end of the current chunk
    #             overlap = chunk[-len(self.PAT.pattern):]
    #             chunk = ''
    #
    #         # Process the remaining chunk and overlap
    #         if chunk or overlap:
    #             pre_subwords = self.PAT.findall(overlap + chunk)
    #             for pretoken in pre_subwords:
    #                 if pretoken not in self.special_tokens:
    #                     pretoken_counts[pretoken] = pretoken_counts.get(pretoken, 0) + 1
    #
    #     print(f"After processing - Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024} MB")
    #     print(f"Number of unique pre-subwords: {len(pretoken_counts)}")
    #     return pretoken_counts

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
    def _calculate_byte_pair_stats(subwords: list[list[bytes]], pretoken_counts: dict[str, int]) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], dict[int, int]]]:
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

        byte_pairs: dict[tuple[bytes, bytes], int] = collections.defaultdict(int)
        token_indices = {}

        for i, word in enumerate(subwords):
            count = pretoken_counts[bytes(b''.join(word)).decode('utf-8')]

            for j in range(len(word) - 1):
                byte_pairs[(word[j], word[j+1])] = byte_pairs.get((word[j], word[j+1]), 0) + count 
                if (word[j], word[j+1]) not in token_indices:
                    token_indices[(word[j], word[j+1])] = {}
                if i not in token_indices[(word[j], word[j+1])]:
                    token_indices[(word[j], word[j+1])][i] = {}
                    token_indices[(word[j], word[j+1])][i]["byte_idx"] = []
                token_indices[(word[j], word[j+1])][i]["count"] = count 
                token_indices[(word[j], word[j+1])][i]["byte_idx"].append(j)

        return byte_pairs, token_indices

    @staticmethod
    def _merge_best_pair(subwords: list[list[bytes]], byte_pairs: dict[tuple[bytes, bytes], int], token_indices, merges: list[tuple[bytes, bytes]], vocab: Vocab) -> None:
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

            # merge
            merged_idxs = byte_idxs
            for merged_idx in merged_idxs:
                if merged_idx > 0: 
                    pair = (subwords[tkn_idx][merged_idx - 1], subwords[tkn_idx][merged_idx])
                    if merged_idx - 1 in token_indices[pair][tkn_idx]["byte_idx"]:
                        token_indices[pair][tkn_idx]["byte_idx"].remove(merged_idx - 1)
                if merged_idx < len(subwords[tkn_idx]) - 2:
                    pair = (subwords[tkn_idx][merged_idx+1], subwords[tkn_idx][merged_idx + 2])
                    if merged_idx + 1 in token_indices[pair][tkn_idx]["byte_idx"]:
                        token_indices[pair][tkn_idx]["byte_idx"].remove(merged_idx + 1)

            for byte_idx in byte_idxs[::-1]:
                subwords[tkn_idx][byte_idx] = new_token
                subwords[tkn_idx].pop(byte_idx + 1)

            merged_idxs = [idx-pos for pos, idx in enumerate(byte_idxs)]

            for merged_idx in merged_idxs:
                count = token_indices[best_pair][tkn_idx]["count"]
                # create new entry for merged token neighbors
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
        

            # update token indices
            for byte_idx in range(len(subwords[tkn_idx]) - 1): 
                pair = (subwords[tkn_idx][byte_idx], subwords[tkn_idx][byte_idx + 1])
                token_indices[pair][tkn_idx]["byte_idx"] = []

            for byte_idx in range(len(subwords[tkn_idx]) - 1):
                pair = (subwords[tkn_idx][byte_idx], subwords[tkn_idx][byte_idx + 1])
                token_indices[pair][tkn_idx]["byte_idx"].append(byte_idx)


        byte_pairs.pop(best_pair)
        token_indices.pop(best_pair)

    @staticmethod
    def _update_byte_pair_stats(subwords: list[list[bytes]], i: int, j: int, new_token: bytes, byte_pairs: dict[tuple[bytes, bytes], int], token_indices, freq: int) -> None:
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
        # Save vocabulary
        with open(vocab_file, 'w', encoding='utf-8') as f:
            vocab_data = {
                'vocab_size': self.vocab_size,
                'special_tokens': list(self.special_tokens),
                'vocab': {}
            }
            for idx, token in self.vocab.get_idx_to_token().items():
                try:
                    decoded_token = token.decode('utf-8')
                except UnicodeDecodeError:
                    decoded_token = f'<byte_{token.hex()}>'
                vocab_data['vocab'][idx] = decoded_token
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

        # Save merges
        with open(merges_file, 'w', encoding='utf-8') as f:
            merges_data = []
            for merge in self.merges:
                try:
                    decoded_merge = (merge[0].decode('utf-8'), merge[1].decode('utf-8'))
                except UnicodeDecodeError:
                    decoded_merge = (f'<byte_{merge[0].hex()}>', f'<byte_{merge[1].hex()}>')
                merges_data.append(decoded_merge)
            json.dump(merges_data, f, ensure_ascii=False, indent=2)
