from functools import lru_cache
from math import inf

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


@lru_cache()
def bytes_to_unicode() -> dict[bytes, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    bs = [bytes([b]) for b in bs]
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


@lru_cache()
def unicode_to_bytes() -> dict[str, bytes]:
    return {c: b for b, c in bytes_to_unicode().items()}


@lru_cache()
def byte_seq_to_str(b: bytes) -> str:
    return "".join(bytes_to_unicode()[bytes([byte])] for byte in b)


@lru_cache()
def str_to_byte_seq(s: str) -> bytes:
    decoder = unicode_to_bytes()
    return b"".join([decoder[c] for c in s])


def merge_tokens_bytes(tokens_bytes: list[bytes], merges_rank: dict[tuple[bytes, bytes], int]):
    result = []
    for token_bytes in tokens_bytes:
        result.append(merge_token_bytes(token_bytes, merges_rank))
    return result


def merge_token_bytes(token_bytes: bytes, merges_rank: dict[tuple[bytes, bytes], int]):
    parts = [bytes([b]) for b in token_bytes]

    while True:
        new_token = []
        pairs = zip(parts, parts[1:])
        if len(parts) < 2:
                return parts
        merge_pair = min(pairs, key=lambda p: merges_rank.get(p, float("inf")))
        if merge_pair not in merges_rank:
            break
        p0, p1 = merge_pair
        i = 0
        while i < len(parts):
            if i < len(parts) - 1 and parts[i] == p0 and parts[i + 1] == p1:
                new_token.append(p0 + p1)
                i += 2
            else:
                new_token.append(parts[i])
                i += 1
        parts = new_token
    return parts
