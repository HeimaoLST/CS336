


from functools import lru_cache


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
@lru_cache()
def bytes_to_unicode() -> dict[bytes, str]:
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n +=1
    bs = [bytes([b]) for b in bs]
    cs = [chr(n) for n in cs]
    return dict(zip(bs,cs))

@lru_cache()
def byte_seq_to_str(b: bytes) -> str:
    return "".join(bytes_to_unicode()[bytes([byte])] for byte in b)

@lru_cache()
def str_to_byte_seq(s: str) -> bytes:
    return bytes([ord(c) for c in s])

def merge_tokens_bytes(tokens_bytes:list[bytes]):
    for token_bytes in tokens_bytes:
        pass

def merge_token_bytes(token_bytes:bytes):
    
