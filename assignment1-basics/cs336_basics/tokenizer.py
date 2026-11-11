import json
import os
from typing import Iterable

import regex

from cs336_basics.utils import PAT, merge_tokens_bytes, str_to_byte_seq



class Tokenizer:
    def __init__(self,vocab:dict[int,bytes],merges:list[tuple[bytes,bytes]],special_tokens:list[str] | None) :
        self.vocab = vocab
        self.bytes_to_id = {tok:tid for tid,tok in vocab.items()}
        self.merges = merges
        self.merges_rank = {m: i for i, m in enumerate(merges)}
        self.special_tokens = special_tokens or []
        for token in self.special_tokens:
            token_bytes = token.encode("UTF-8")
            if token_bytes not in self.bytes_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.bytes_to_id[token_bytes] = new_id
        self.text_pattern = regex.compile(PAT)
        if not special_tokens:
            self.special_pattern = None
            return
        escaped_tokens = sorted(
                special_tokens,
                key=len,
                reverse=True
            )
        special_pattern = "("+"|".join([regex.escape(tok) for tok in  escaped_tokens]) + ")"
        self.special_pattern = regex.compile(special_pattern)

    @classmethod
    def from_file(cls, vocab_path:str | os.PathLike, merges_path:str | os.PathLike, special_tokens: list[str] | None):
        with open(vocab_path, encoding="UTF-8") as f:
            vocab = json.load(f)
        vocab = {tid:str_to_byte_seq(tok) for tok, tid in vocab.items()}
        with open(merges_path, encoding="UTF-8") as f:
            merges = json.load(f)
        merges = [(str_to_byte_seq(i), str_to_byte_seq(j)) for i, j in merges]
        return cls(vocab, merges, special_tokens)

    def _encode_chunk(self,text:str) -> list[int]:
        tokens = self.text_pattern.findall(text)
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
    def encode(self,text:str):
        if not self.special_pattern:
            return self._encode_chunk(text)
        parts = self.special_pattern.split(text)
        ids = []
        for part in parts:
            if part not in self.special_tokens:
                ids.extend(self._encode_chunk(part))
            else:
                ids.append(self.bytes_to_id.get(part.encode("UTF-8")))
        return ids

    def encode_iterable(self,iterable:Iterable[str]) -> Iterable[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self,ids:list[int]) -> str:
        all_bytes = b"".join([self.vocab.get(id) for id in ids])
        return all_bytes.decode("UTF-8",errors="replace")
