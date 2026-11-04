import os
import time
import regex as re
from collections import Counter, defaultdict
import json

from cs336_basics.pretokenization_example import parallelize_preprocess
from cs336_basics.utils import bytes_to_unicode,byte_seq_to_str
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


def find_best_pair(pair_counter: Counter):
    pair,_ = max(pair_counter.items(),key = lambda item : (item[1],item[0]))
    return pair



def update_pair_counter(pair_counter, best_pair, pair_index: dict, word_freq):
    affected_words = list(pair_index.pop(best_pair, ()))
    for word in affected_words:
        freq = word_freq.pop(word, None)
        if freq is None:
            continue
        for pair in zip(word, word[1:]):
            pair_counter[pair] -= freq
            if pair_counter[pair] <= 0:
                del pair_counter[pair]
        new_word = merge_word(word, best_pair)
        word_freq[new_word] = word_freq.get(new_word, 0) + freq
        for pair in zip(new_word, new_word[1:]):
            pair_counter[pair] += freq
            pair_index.setdefault(pair, set()).add(new_word)


def update_vocabulary_and_merges(best_pair, vocabulary, merges):
    p0, p1 = best_pair
    merged_token = p0 + p1
    vocabulary.append(merged_token)
    # merges[(p0.decode("latin-1"), p1.decode("latin-1"))] = len(vocabulary) - 1
    merges.append((p0, p1))




def init_vocabulary():
    vocabulary = []
    for i in range(256):
        vocabulary.append(bytes([i]))
    return vocabulary


def init_pair_counter_and_pair_index(word_freq):
    pair_counter = Counter()
    pair_index = defaultdict(set)
    for word, cnt in word_freq.items():
        for p0, p1 in zip(word, word[1:]):
            pair_counter[(p0, p1)] += cnt
            pair_index_set = pair_index.get((p0, p1), set())
            pair_index_set.add(word)
            pair_index[(p0, p1)] = pair_index_set
    return pair_counter, pair_index



def train_BPE(filepath: str | os.PathLike, vocabulary_size: int, special_token: list[str] | None):
    preprocess_start = time.perf_counter()
    word_freq = parallelize_preprocess(filepath, special_token, num_processes=4)
    preprocess_end = time.perf_counter()
    print(f"Preprocess time: {preprocess_end - preprocess_start}s")
    vocabulary = init_vocabulary()
    merges = []
    loop_start = time.perf_counter()
    pair_counter, pair_index = init_pair_counter_and_pair_index(word_freq)
    while len(vocabulary) < vocabulary_size:
        if len(vocabulary) % 1000 == 0:
            step_time = time.perf_counter()
            step_spent = step_time - loop_start
            print(
                f"Vocabulary size is {len(vocabulary)} spent time {step_spent}s rate: {len(vocabulary) / step_spent} "
            )
        best_pair = find_best_pair(pair_counter)
        update_vocabulary_and_merges(best_pair, vocabulary, merges)
        update_pair_counter(pair_counter, best_pair, pair_index, word_freq)
    if special_token is not None:
        for token in special_token:
            if token not in vocabulary:
                vocabulary.append(token.encode("utf-8"))
    vocabulary_dict:dict[int,bytes] = {i: b for i, b in enumerate(vocabulary)}
    # formatted_merges = {f"{k[0]} {k[1]}": v for k, v in merges.items()}
    # with open("vocabulary.json", mode="w", encoding="UTF-8") as f:
    #     json.dump(vocabulary_dict, f, indent=4, ensure_ascii=False)
    # with open("merges.json", mode="w", encoding="UTF-8") as f:
    #     json.dump(formatted_merges, f, indent=4, ensure_ascii=False)
    # ith open("vocabulary_5k.json", mode="w", encoding="UTF-8") as f:
    #     json.dump(vocabulary_dict, f, indent=4, ensure_ascii=False)
    # with open("merges_5k.json", mode="w", encoding="UTF-8") as f:
    #     json.dump(formatted_merges, f, indent=4, ensure_ascii=False)
    return vocabulary_dict, merges

if __name__ == "__main__":
    ## Usage
    filepath = "data/TinyStoriesV2-GPT4-train.txt"
    special_token = ["<|endoftext|>"]
    vocab, merges = train_BPE(filepath, 10_000, special_token)
    bytes_to_unicode_map = bytes_to_unicode()
    vocab_with_unicode = {byte_seq_to_str(v): i for i, v in vocab.items()}
    merges_with_unicode = [(byte_seq_to_str(k[0]), byte_seq_to_str(k[1])) for k in merges]
    with open("vocabulary_10k.json", mode="w", encoding="UTF-8") as f:
        json.dump(vocab_with_unicode, f, indent=4, ensure_ascii=False)
    with open("merges_10k.json", mode="w", encoding="UTF-8") as f:
        json.dump(merges_with_unicode, f, indent=4, ensure_ascii=False)
