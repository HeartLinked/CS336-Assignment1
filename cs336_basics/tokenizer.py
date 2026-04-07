import regex as re
import os

from collections import Counter

from cs336_basics.pretokenization_example import find_chunk_boundaries
import multiprocessing

def process_chunk(
    task: tuple[int, int],
    input_path: str | os.PathLike,
    pattern_pre: str, 
    pattern: str
) -> dict[tuple[bytes, ...]] :
    with open(input_path, "rb") as f:
        f.seek(task[0])
        chunk: str = f.read(task[1] - task[0]).decode("utf-8", errors="ignore")
    # Removing special tokens before pre-tokenization
    pieces: list[str] = re.split(pattern_pre, chunk)

    pretoken_counts:dict[tuple[bytes, ...]] = Counter()
    BYTE_TOKENS = tuple(bytes([i]) for i in range(256))

    for piece in pieces:
        # every piece is a long text with space, we need to use pattern to spilt
        if not piece:
            continue
        word = re.finditer(pattern, piece)
        for w in word: 
            pretoken = w.group(0)
            pretoken_bytes = tuple(BYTE_TOKENS[b] for b in pretoken.encode("utf-8"))
            pretoken_counts[pretoken_bytes] += 1
    return pretoken_counts

def get_pair_counts(
    pre_token: dict[tuple[bytes, ...]]
) -> dict[tuple[bytes, ...]]:
    pair_counts: dict[tuple[bytes, ...]] = Counter()
    for token_seq, freq in pre_token.items():
        # token_seq is like (b'n', b'e', b'w', b'e', b's', b't')
        for a, b in zip(token_seq, token_seq[1:]):
            pair_counts[(a, b)] += freq
    return pair_counts

def merge_pretoken(
    token_seq: tuple[bytes, ...],
    top_pair: tuple[bytes, bytes],
) -> tuple[bytes, ...]:
    merged = []
    i = 0
    while i < len(token_seq):
        if i + 1 < len(token_seq) and (token_seq[i], token_seq[i + 1]) == top_pair:
            merged.append(token_seq[i] + token_seq[i + 1])
            i += 2
        else:
            merged.append(token_seq[i])
            i += 1
    return tuple(merged)

def my_run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    pattern_pre = "|".join(re.escape(tok) for tok in special_tokens)
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # return value
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []
    next_id = 0

    # add special tokens to vocab
    for special_token in special_tokens:
        vocab[next_id] = special_token.encode("utf-8")
        next_id += 1

    for i in range(256):
        vocab[next_id] = bytes([i])
        next_id += 1

    num_processes = kwargs.get("num_processes", 1)

    if num_processes == 1:
        file_size = os.path.getsize(input_path)
        results = [process_chunk((0, file_size), input_path, pattern_pre, pattern)]
    else:
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        tasks = [(start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.starmap(
                process_chunk,
                [(task, input_path, pattern_pre, pattern) for task in tasks],
            )
    
    # result: list[dict[tuple[bytes, ...]]]
    # print("Token counts for each chunk:", len(results))
    pretoken_counts: Counter[tuple[bytes, ...]] = Counter()
    for result in results:
        pretoken_counts.update(result)

    t = max(0, vocab_size - 256 - len(special_tokens))
    for i in range(t):
        pair_counts: dict[tuple[bytes, ...]] = get_pair_counts(pretoken_counts)
        top_pair: tuple[bytes, bytes] = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        merges.append(top_pair)
        new_token = top_pair[0] + top_pair[1]
        vocab[next_id] = new_token
        next_id += 1

        updated_counts: Counter[tuple[bytes, ...]] = Counter()
        for token_seq, freq in pretoken_counts.items():
            merged_seq = merge_pretoken(token_seq, top_pair)
            updated_counts[merged_seq] += freq
        pretoken_counts = updated_counts

    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    return vocab, merges
    # raise NotImplementedError

if __name__ == '__main__':
    print("--run with main--")
    # run_train_bpe("/Users/bytedance/Documents/Dev/learn-daft/py/data/TinyStoriesV2-GPT4-valid.txt", 200, ["<|endoftext|>"])
    my_run_train_bpe("/Users/bytedance/Documents/Dev/learn-daft/py/data/s1.txt", 261, ["<|endoftext|>"])

