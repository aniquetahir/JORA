from functools import partial
import jax
from jax import Array
import jax.random as rand
import jax.numpy as jnp
from typing import Optional

from ..rand_utils import split_key_nullable
from ..tree_utils import stack_leaves
from .ModelConfig import ModelConfig
from .decoder_block import DecoderBlock, DecoderBlock as Decoder, \
    check_decoder_block, decoder_block, decoder_block_lora, init_decoder_block

def check_decoder(params: Decoder, *, model_config: ModelConfig) -> None:
    def inner(state, input_):
        assert isinstance(input_, DecoderBlock)
        check_decoder_block(input_, model_config=model_config)
        return None, None
    jax.lax.scan(inner, None, params)

def init_decoder(*, key: Array, model_config: ModelConfig) -> Decoder:
    return stack_leaves([init_decoder_block(key=subkey, model_config=model_config) for subkey in rand.split(key, num=model_config.n_layers)])

@partial(jax.jit, static_argnames=('model_config',))
def decoder(params: Decoder, seq: Array, attn_mask: Array, *, key: Optional[Array], model_config: ModelConfig) -> Array:
    def inner(state, input_):
        key, seq = state
        key, subkey = split_key_nullable(key)
        seq = decoder_block(input_, seq, attn_mask, key=subkey, model_config=model_config)
        return (key, seq), None
    (key, seq), _ = jax.lax.scan(inner, (key, seq), params)
    return seq


@partial(jax.jit, static_argnames=('lora_config', 'model_config',))
def decoder_lora(lora_params, lora_config, params: Decoder, seq: Array, attn_mask: Array, *, key: Optional[Array], model_config: ModelConfig) -> Array:
    # get the lora parameters for each block
    def inner(state, input_):


        key, seq, i = state
        key, subkey = split_key_nullable(key)
        lora_i = {
            'q_lora_A': lora_params['q_lora_A'][i],
            'q_lora_B': lora_params['q_lora_B'][i],
            'v_lora_A': lora_params['v_lora_A'][i],
            'v_lora_B': lora_params['v_lora_B'][i],
        }
        seq = decoder_block_lora(lora_i, lora_config, input_, seq, attn_mask, key=subkey, model_config=model_config)
        return (key, seq, i+1), None
    (_, seq, _), _ = jax.lax.scan(inner, (key, seq, 0), params)
    return seq
