from models.tokenizer.bpe_tokenizer import BPETokenizer 

def run_train_bpe(input_path, vocab_size, special_tokens):
    tokenizer = BPETokenizer(vocab_size, special_tokens)
    vocab, merges = tokenizer.from_file(input_path)

    return vocab, merges

path = "fixtures/corpus.en"
vocab_size = 10000
special_tokens = ["<|endoftext|>"]

run_train_bpe(path, vocab_size, special_tokens)
