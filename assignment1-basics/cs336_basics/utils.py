


from functools import lru_cache


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
