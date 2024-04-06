class Vocab:
    def __init__(self, special_tokens: list[str] = []) -> None:
        self.token_to_idx: dict[bytes, int] = {} 
        self.idx_to_token: dict[int, bytes] = {}

        for token in special_tokens:
            self.add_token(token.encode("utf-8"))

        # init 256 possible byte values
        for i in range(256):
            self.add_token(bytes([i]))
           
        self.unk_idx: int = 0

    def __len__(self) -> int:
        return len(self.idx_to_token)

    def __getitem__(self, token: bytes) -> int:
        return self.token_to_idx.get(token, self.unk_idx)

    def __contains__(self, token: bytes) -> bool:
        return token in self.token_to_idx
    
    def add_token(self, token: bytes) -> int:
        if token not in self.token_to_idx:
            self.token_to_idx[token] = len(self.idx_to_token)
            self.idx_to_token[len(self.idx_to_token)] = token

        return self.token_to_idx[token]
    
    def get_idx_to_token(self) -> dict[int, bytes]:
        return self.idx_to_token

    def set_unk_idx(self, unk_idx: int) -> None:
        self.unk_idx = unk_idx

