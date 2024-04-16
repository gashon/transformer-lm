import cProfile
import pstats
import io
import time
from memory_profiler import profile
from models.tokenizer.bpe_tokenizer import BPETokenizer
from models.tokenizer.util import train_bpe


# @profile
def profile_train_bpe(input_path, vocab_size, special_tokens, name):
    pr = cProfile.Profile()
    pr.enable()

    tokenizer = BPETokenizer(vocab_size, special_tokens)

    start_time = time.time()
    # vocab, merges = tokenizer.from_file(input_path)
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    end_time = time.time()

    pr.disable()

    tokenizer.save_to_file(f"data/tkn/{name}-vocab.json", f"data/tkn/{name}-merge.json")

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("time")
    ps.print_stats()
    print(s.getvalue())

    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")

    return vocab, merges
