from typing import NamedTuple, Optional


class GemmaConfig(NamedTuple):
    n_heads: int
    n_kv: int


GemmaConfig2B = GemmaConfig(8, 1)
GemmaConfig7B = GemmaConfig(16, 16)

GEMMA_VERSIONS = set([
    '2b',
    '2b-it',
    '7b',
    '7b-it',
    '1.1-2b-it',
    '1.1-7b-it'
])