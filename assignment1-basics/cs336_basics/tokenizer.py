import os
import json
import regex as re

from cs336_basics.utils import str_to_byte_seq
from cs336_basics.utils import PAT
from cs336_basics.utils import merge_tokens_bytes
class Tokenizer:
    def __init__(self,vocab:dict[int,bytes],merges:list[tuple[bytes,bytes]],special_tokens:list[str] | None) :
        self.vocab = vocab
        self.bytes_to_id = {tok:tid for tid,tok in vocab.items()}
        self.merges = merges
        self.merges_rank = {m: i for i, m in enumerate(merges)}
        self.special_tokens = special_tokens
    @classmethod
    def from_file(cls, vocab_path:str | os.PathLike, merges_path:str | os.PathLike, special_tokens: list[str] | None):
        with open(vocab_path, encoding="UTF-8") as f:
            vocab = json.load(f)
        vocab = {tid:str_to_byte_seq(tok) for tok, tid in vocab.items()}
        with open(merges_path, encoding="UTF-8") as f:
            merges = json.load(f)
        merges = [(str_to_byte_seq(i), str_to_byte_seq(j)) for i, j in merges]
        print(merges[:100])
        return cls(vocab, merges, special_tokens)

    def encode(self,text:str) -> list[int]:
       tokens = re.findall(PAT, text)
       tokens_bytes = [t.encode("UTF-8") for t in tokens]
       ##merge
       merged_tokens_bytes = merge_tokens_bytes(tokens_bytes, self.merges_rank)
       ##convert to vocab ids
       ids = []
       # print(merged_tokens_bytes)
       for token in merged_tokens_bytes:
           for byte in token:
               ids.append(self.bytes_to_id[byte])
       return ids
