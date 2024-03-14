import einops as op
from functools import partial
import jax
from jax import Array
import jax.nn as nn
import jax.numpy as jnp
import jax.random as rand
import math
from typing import Any, NamedTuple

from .ModelConfig import ModelConfig
from .rotary_embedding import rotary_embedding

class Attention(NamedTuple):
    q_proj: Any  # Array
    k_proj: Any  # Array
    v_proj: Any  # Array
    out_proj: Any  # Array

# class LoraAttention(NamedTuple):
#     q_proj: Any  # Array
#     k_proj: Any  # Array
#     v_proj: Any  # Array
#     out_proj: Any  # Array
#     q_proj_A: Any  # Array
#     v_proj_A: Any  # Array
#     q_proj_B: Any  # Array
#     v_proj_B: Any  # Array
#
#     def __init__(self, loraConfig, attention: Attention):
#         self.q_proj = attention.q_proj
#         self.k_proj = attention.k_proj
#         self.v_proj = attention.v_proj
#         self.out_proj = attention.out_proj
#         self.loraConfig = loraConfig
#         q_proj_shape, v_proj_shape = attention.q_proj.shape, attention.v_proj.shape
#         DB, H, N_REP, N_HEADS, D_K = q_proj_shape
#         assert v_proj_shape == (DB, H, N_HEADS, D_K,)
#         key, split_qa, split_va = rand.split(loraConfig.KEY, 3)
#         self.q_proj_A = rand.normal(split_qa, (DB, H, N_REP, N_HEADS, loraConfig.LORA_R))
#         self.v_proj_A = rand.normal(split_va, (DB, H, N_HEADS, loraConfig.LORA_R))
#         self.q_proj_B = jnp.zeros((DB, loraConfig.LORA_R, N_REP, N_HEADS, D_K))
#         self.v_proj_B = jnp.zeros((DB, loraConfig.LORA_R, N_HEADS, D_K))
#

def check_attention(params: Attention, *, model_config: ModelConfig) -> None:
    assert isinstance(params.q_proj, Array)
    assert isinstance(params.k_proj, Array)
    assert isinstance(params.v_proj, Array)
    assert isinstance(params.out_proj, Array)

    assert params.q_proj.shape == (model_config.d_model, model_config.n_rep_kv, model_config.n_heads_kv, model_config.d_k)
    assert params.k_proj.shape == (model_config.d_model, model_config.n_heads_kv, model_config.d_k)
    assert params.v_proj.shape == (model_config.d_model, model_config.n_heads_kv, model_config.d_v)
    assert params.out_proj.shape == (model_config.n_rep_kv, model_config.n_heads_kv, model_config.d_v, model_config.d_model)

def init_attention(*, key, model_config: ModelConfig) -> Attention:
    upper = 1. / math.sqrt(model_config.d_model)
    key0, key1, key2, key3 = rand.split(key, num=4)
    q_proj = rand.truncated_normal(key0, -upper, upper, (model_config.d_model, model_config.n_rep_kv, model_config.n_heads_kv, model_config.d_k))
    k_proj = rand.truncated_normal(key1, -upper, upper, (model_config.d_model, model_config.n_heads_kv, model_config.d_k))
    v_proj = rand.truncated_normal(key2, -upper, upper, (model_config.d_model, model_config.n_heads_kv, model_config.d_v))
    out_proj = rand.truncated_normal(key3, -upper, upper, (model_config.n_rep_kv, model_config.n_heads_kv, model_config.d_v, model_config.d_model))
    return Attention(q_proj, k_proj, v_proj, out_proj)

@partial(jax.jit, static_argnames=('model_config',))
def attention(params: Attention, src_seq: Array, dst_seq: Array, attn_mask: Array, *, model_config: ModelConfig) -> Array:
    q = op.einsum(src_seq, params.q_proj, 'batch_size src_seq_len d_model, d_model n_rep_kv n_heads_kv d_k -> batch_size n_rep_kv n_heads_kv src_seq_len d_k')
    k = op.einsum(dst_seq, params.k_proj, 'batch_size dst_seq_len d_model, d_model n_heads_kv d_k -> batch_size n_heads_kv dst_seq_len d_k')
    v = op.einsum(dst_seq, params.v_proj, 'batch_size dst_seq_len d_model, d_model n_heads_kv d_v -> batch_size n_heads_kv dst_seq_len d_v')

    q = rotary_embedding(q)
    k = rotary_embedding(k)

    qk = op.einsum(q, k, 'batch_size n_rep_kv n_heads_kv src_seq_len d_k, batch_size n_heads_kv dst_seq_len d_k -> batch_size n_rep_kv n_heads_kv src_seq_len dst_seq_len')
    qk /= math.sqrt(model_config.d_k)
    qk = jnp.where(attn_mask, qk, -jnp.inf)
    qk = nn.softmax(qk)
    qk = jnp.where(attn_mask, qk, 0)

    qkv = op.einsum(qk, v, 'batch_size n_rep_kv n_heads_kv src_seq_len dst_seq_len, batch_size n_heads_kv dst_seq_len d_v -> batch_size n_rep_kv n_heads_kv src_seq_len d_v')

    out = op.einsum(qkv, params.out_proj, 'batch_size n_rep_kv n_heads_kv src_seq_len d_v, n_rep_kv n_heads_kv d_v d_model -> batch_size src_seq_len d_model')
    return out



@partial(jax.jit, static_argnames=('lora_config', 'model_config',))
def attention_lora(lora_params, lora_config, params: Attention, src_seq: Array, dst_seq: Array, attn_mask: Array, *, model_config: ModelConfig) -> Array:
    lora_r = lora_config.LORA_R
    lora_alpha = lora_config.LORA_ALPHA

    q_AB = (lora_alpha/lora_r) * op.einsum(
        lora_params['q_lora_A'],
        lora_params['q_lora_B'],
        'h n_rep n_heads r, r n_rep n_heads d_k -> h n_rep n_heads d_k')

    v_AB = (lora_alpha/lora_r) * op.einsum(
        lora_params['v_lora_A'],
        lora_params['v_lora_B'],
        'h n_heads r, r n_heads d_k -> h n_heads d_k')

    q_proj = params.q_proj + q_AB
    v_proj = params.v_proj + v_AB

    q = op.einsum(src_seq, q_proj, 'batch_size src_seq_len d_model, d_model n_rep_kv n_heads_kv d_k -> batch_size n_rep_kv n_heads_kv src_seq_len d_k')
    k = op.einsum(dst_seq, params.k_proj, 'batch_size dst_seq_len d_model, d_model n_heads_kv d_k -> batch_size n_heads_kv dst_seq_len d_k')
    v = op.einsum(dst_seq, v_proj, 'batch_size dst_seq_len d_model, d_model n_heads_kv d_v -> batch_size n_heads_kv dst_seq_len d_v')

    q = rotary_embedding(q)
    k = rotary_embedding(k)

    qk = op.einsum(q, k, 'batch_size n_rep_kv n_heads_kv src_seq_len d_k, batch_size n_heads_kv dst_seq_len d_k -> batch_size n_rep_kv n_heads_kv src_seq_len dst_seq_len')
    qk /= math.sqrt(model_config.d_k)
    qk = jnp.where(attn_mask, qk, -jnp.inf)
    qk = nn.softmax(qk)
    qk = jnp.where(attn_mask, qk, 0)

    qkv = op.einsum(qk, v, 'batch_size n_rep_kv n_heads_kv src_seq_len dst_seq_len, batch_size n_heads_kv dst_seq_len d_v -> batch_size n_rep_kv n_heads_kv src_seq_len d_v')

    out = op.einsum(qkv, params.out_proj, 'batch_size n_rep_kv n_heads_kv src_seq_len d_v, n_rep_kv n_heads_kv d_v d_model -> batch_size src_seq_len d_model')
    return out
