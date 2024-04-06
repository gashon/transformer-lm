import time
from memory_profiler import profile
from models.tokenizer.bpe_tokenizer import BPETokenizer

@profile
def run_train_bpe(input_path, vocab_size, special_tokens):
    tokenizer = BPETokenizer(vocab_size, special_tokens)

    start_time = time.time()
    vocab, merges = tokenizer.from_file(input_path)
    end_time = time.time()

    tokenizer.save_to_file("output/vocab.json", "output/merge.json")

    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")

    return vocab, merges

path = "tests/fixtures/corpus.en"
vocab_size = 500 
special_tokens = ["<|endoftext|>"]

run_train_bpe(path, vocab_size, special_tokens)
