
from cs336_basics.tokenizer import Tokenizer



# def test_merge_pair():
    # assert merge_pair(['a','b','c','b','c'],('b','c')) == ['a', 'bc','bc']
def test_encoding():
    tokenizer = Tokenizer.from_file("vocabulary_10k.json", "merges_10k.json", None)
    print(tokenizer.encode("hello world"))
    assert True
