from itertools import repeat

import jax
import jax.random as rand
from typing import Optional


def split_key_nullable(key: Optional[jax.Array], num: int=2):
    if key is None:
        return tuple(repeat(None, num))
    else:
        return rand.split(key, num)
