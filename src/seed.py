"""seed.py — one stable, collision-resistant seed function, shared by every
synthetic data generator in this project so the same ticker always gets the
same fake data within a run (and so AAPL and any other ticker never collide,
unlike a naive sum-of-character-codes seed would)."""

import zlib


def stable_seed(s: str) -> int:
    return zlib.crc32(s.encode()) % (2**32)
