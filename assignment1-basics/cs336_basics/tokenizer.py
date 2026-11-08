import os
import json
import re
from cs336_basics.utils import str_to_byte_seq
from utils import PAT

class Tokenizer:
    def __init__(self,vocab:dict[int,bytes],merges:list[tuple[bytes,bytes]],special_tokens:list[str] | None) :
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
    @classmethod
    def from_file(cls, vocab_path:str | os.PathLike, merges_path:str | os.PathLike, special_tokens: list[str] | None):
        with open(vocab_path, mode="r", encoding="UTF-8") as f:
            vocab = json.load(f)
        vocab = {v: str_to_byte_seq(i) for i, v in vocab.items()}
        with open(merges_path, mode="r", encoding="UTF-8") as f:
            merges = json.load(f)
        merges = [(str_to_byte_seq(i), str_to_byte_seq(j)) for i, j in merges]
        return cls(vocab, merges, special_tokens)
    def encode(self,text:str) -> list[int]:
       tokens = re.findall(PAT, text)
       tokens_bytes = [str_to_byte_seq(t) for t in tokens]
       ##merge
       merged_tokens_bytes = 
