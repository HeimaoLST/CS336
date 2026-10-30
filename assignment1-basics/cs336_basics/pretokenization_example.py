from functools import cache, lru_cache
import json
import os
import time
from collections import Counter
from multiprocessing import Pool
from typing import BinaryIO

import regex as re


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def merge_word(tokens, pair):
    a, b = pair
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
            new_tokens.append(a + b)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return tuple(new_tokens)


def merge_pair(word_dict, pair):
    merge_pair_start = time.perf_counter()
    new_word_dict = {}
    for tokens, cnt in word_dict.items():
        new_tokens = merge_word(tokens, pair)
        new_word_dict[new_tokens] = new_word_dict.get(new_tokens, 0) + cnt
    merge_pair_end = time.perf_counter()
    # print(f"merge_pair_time : {merge_pair_end - merge_pair_start}")
    return new_word_dict


def find_best_pair(word_dict: dict):
    pair_counter = Counter()
    for word, cnt in word_dict.items():
        for p0, p1 in zip(word, word[1:]):
            pair_counter[(p0, p1)] += cnt
    return pair_counter.most_common(1)[0][0]


def train_BPE(filepath: str, vocabulary_size: int, special_token: list[str]):
    word_freq = parallelize_preprocess(filepath, special_token, num_processes=4)
    vocabulary = init_vocabulary(special_token)
    merges = {}
    merge_index = 0
    loop_start = time.perf_counter()
    while len(vocabulary) < vocabulary_size:
        if len(vocabulary) % 1000 == 0:
            step_time = time.perf_counter()
            step_spent = step_time - loop_start
            print(
                f"Vocabulary size is {len(vocabulary)} spent time {step_spent}s rate: {len(vocabulary) / step_spent} "
            )
        best_pair = find_best_pair(word_freq)
        p0, p1 = best_pair
        merged_token = p0 + p1
        vocabulary.append(merged_token)
        merges[(p0.decode("latin-1"), p1.decode("latin-1"))] = merge_index
        merge_index += 1
        word_freq = merge_pair(word_freq, best_pair)
    vocabulary_dict = {i: b.decode("latin-1") if isinstance(b, bytes) else b for i, b in enumerate(vocabulary)}
    formatted_merges = {f"{k[0]} {k[1]}": v for k, v in merges.items()}
    with open("vocabulary.json", mode="w", encoding="UTF-8") as f:
        json.dump(vocabulary_dict, f, indent=4, ensure_ascii=False)
    with open("merges.json", mode="w", encoding="UTF-8") as f:
        json.dump(formatted_merges, f, indent=4, ensure_ascii=False)


def parallelize_preprocess(filepath, special_token, num_processes):
    # special_token = ["<|endoftext|>"]
    with open(filepath, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    reduces = Counter()
    with Pool(processes=4) as pool:
        aync_result = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            res = pool.apply_async(split_text, args=(filepath, special_token, start, end))
            aync_result.append(res)
        result = [res.get() for res in aync_result]
        for counter in result:
            reduces.update(counter)
    word_freq: dict[tuple[bytes, ...], int] = {
        tuple(bytes([b]) for b in word.encode("UTF-8")): count for word, count in reduces.items()
    }
    return word_freq


def split_text(path, special_token, start, end):
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        pattern = "|".join(map(re.escape, special_token))
        # print("split_pattern:", pattern)
        chunk_splits = re.split(pattern, chunk)
        text = "".join(chunk_splits)
        pre_token = re.finditer(PAT, text)
        token_counts = Counter(m.group(0) for m in pre_token)
        return token_counts


def init_vocabulary(special_token: list[str]):
    vocabulary = []
    vocabulary += special_token
    for i in range(256):
        vocabulary.append(bytes([i]))
    return vocabulary


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

if __name__ == "__main__":
    ## Usage
    filepath = "../data/TinyStoriesV2-GPT4-valid.txt"
    num_processes = 4
    special_token = ["<|endoftext|>"]
    train_BPE(filepath, 10_000, special_token)
