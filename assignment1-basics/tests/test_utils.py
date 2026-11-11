
from cs336_basics.tokenizer import Tokenizer

import pytest
TEXTS = [
        "hello world",
        "",                          # empty string
        "héllo 世界 😀",              # multi-byte UTF-8
        "  leading spaces",
        "trailing spaces   ",
        "line1\n\nline2\ttab",       # whitespace mix
        "don't you're I'll",         # the 's 're 'll rules in PAT
        "12345 3.14",                # numbers
        "zxqvj qwrtp",               # rare words, so few merges apply
    ]
# def test_merge_pair():
    # assert merge_pair(['a','b','c','b','c'],('b','c')) == ['a', 'bc','bc']
def test_encoding():
    tokenizer = Tokenizer.from_file("vocabulary_10k.json", "merges_10k.json", None)
    encode = tokenizer.encode("hello world")
    print(encode)
    decode = tokenizer.decode(encode)
    print(decode)

    assert decode =="hello world"


@pytest.mark.parametrize("text", TEXTS)
def test_roundtrip(text):
    tok = Tokenizer.from_file("vocabulary_10k.json", "merges_10k.json", None)
    ids = tok.encode(text)
    assert tok.decode(ids) == text
