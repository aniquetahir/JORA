from typing import NamedTuple
import os
from collections import defaultdict
import sys
from os.path import join as pjoin
from pathlib import Path
import tree

import jax
from jax import Array
from jax.experimental.multihost_utils import process_allgather
import jax.numpy as jnp
import jax.random as rand

# Define the mesh
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils

import math
import optax
import time
from transformers import LlamaTokenizer
from tqdm import tqdm
from typing import Any, Callable, Optional, Dict

import sys
import jora.lib
sys.modules['lib'] = jora.lib

from lib.dataloader import LlamaDataLoader
from lib.alpaca_data import AlpacaDataset, TrainData, alpaca_collate_fn_train
from lib.loss import cross_entropy_loss

from lib.param_utils import load_params, save_params
from jora.hf.hf_to_jax import hf_to_jax
from os.path import join as pjoin

from jax.sharding import Mesh, NamedSharding, PartitionSpec, PositionalSharding
from functools import partial
from collections import namedtuple
import pickle


# gemma imports
from gemma_utils import GemmaTokenizer, get_attention_mask_and_positions
from gemma_config import GemmaConfig, GemmaConfig2B, GemmaConfig7B, GEMMA_VERSIONS
from gemma import params as params_lib
from gemma import sampler as sampler_lib
from gemma import transformer as transformer_lib
import sentencepiece as spm


optimize: Optional[Callable]
active_model_config: GemmaConfig

model_config_mapping = {
    '2b': GemmaConfig2B,
    '2b-it': GemmaConfig2B,
    '7b': GemmaConfig7B,
    '7b-it': GemmaConfig7B
}

cpu_devices = jax.devices('cpu')
cpu_sharding = NamedSharding(Mesh(cpu_devices, ('D',)), PartitionSpec(None))

gpu_devices = jax.devices('gpu')
gpu_sharding_mp = PositionalSharding(gpu_devices)
gpu_sharding_mp = gpu_sharding_mp.reshape((1, len(gpu_devices)))


class ParagemmaConfig(NamedTuple):
    GEMMA_MODEL_PATH: str # e.g. '/tmp/llama2-13B'
    MODEL_VERSION: str # '2b', '7b', '2b-it', '7b-it'
    NUM_SHARDS: int = None
    LORA_R: int = 16
    LORA_ALPHA: int = 16
    LORA_DROPOUT: float = 0.05
    LR: float = 0.0001
    BATCH_SIZE: int = 1
    N_ACCUMULATION_STEPS: int = 8
    MAX_SEQ_LEN = 2000
    N_EPOCHS: int = 7
    SEED: int = 420
    CACHE_SIZE: int = 30 # Numbber of steps in the transformer's cache

is_process_0 = jax.process_index() == 0
cpu_device = jax.devices('cpu')[0]
gpu_device = jax.devices('gpu')[0]


def merge_lora_params(params, lora_params: Dict):
    def merge_fn(path, v):
        # h - num heads, m - model dim, r - lora dim, k - key dim, v - value dim
        if 'kv_einsum' in path:
            v_lora_A = lora_params[path]['v_lora_A']
            v_lora_B = lora_params[path]['v_lora_B']
            merged_V = v[1] + jnp.einsum('hmr,hrk->hmk', v_lora_A, v_lora_B) # this gives us the same dimension as v[1]
            return jnp.vstack([v[0], merged_V])
        elif 'q_einsum' in path:
            return v + jnp.einsum('hmr,hrv->hmv', lora_params[path]['q_lora_A'], lora_params[path]['q_lora_B'])
        else:
            return v

    merged_params = jax.tree_util.tree_map_with_path(merge_fn, params)
    return merged_params


def generate_alpaca_dataset(path: str, split: str, config: ParagemmaConfig, split_percentage=0.8, alpaca_mix=0.3):
    """
    Generate an AlpacaDataset object
    :param path:
    :param split: 'train' or 'test'
    :param config:
    :param split_percentage:
    :param alpaca_mix:
    :return:
    """
    vocab_path = os.path.join(config.GEMMA_MODEL_PATH, 'tokenizer.model')
    vocab = spm.SentencePieceProcessor(vocab_path)
    vocab.Load(vocab_path)

    tokenizer = GemmaTokenizer(vocab)
    dataset = AlpacaDataset(split=split, path=path, split_percentage=split_percentage, tokenizer=tokenizer, alpaca_mix=alpaca_mix)
    return dataset


def train_lora(config: ParagemmaConfig, train_dataset: AlpacaDataset, checkpoint_dir: str, checkpoint_prefix='jax_lora',
               verbose=True) -> None:
    global optimize
    global active_model_config

    if not config.GEMMA_MODEL_PATH or not config.MODEL_VERSION:
        raise ValueError('Please provide a valid GEMMA_MODEL_PATH and MODEL_VERSION in the config (2b, 7b, 2b-it, 7b-it)')

    if not config.MODEL_VERSION in GEMMA_VERSIONS:
        raise ValueError(f'Please provide a valid MODEL_SIZE in the config ({" ,".join(GEMMA_VERSIONS)})')

    # IMPORTANT: these values should be set before jit compilation occurs. Otherwise, it will compile with the wrong values
    active_model_config = model_config_mapping[config.MODEL_VERSION]

    # check if the checkpoint directory exists and create it if it doesn't
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    key = rand.PRNGKey(config.SEED)
    vocab_path = os.path.join(config.GEMMA_MODEL_PATH, 'tokenizer.model')
    vocab = spm.SentencePieceProcessor(vocab_path)
    vocab.Load(vocab_path)
    tokenizer = GemmaTokenizer(vocab)


    # dataset = AlpacaDataset(split='train', path='./merged_dataset_insta_4chan.json', tokenizer=tokenizer, max_len=max_len, alpaca_mix=0.)
    dataset = train_dataset
    collate_fn = partial(alpaca_collate_fn_train, tokenizer, config.MAX_SEQ_LEN)

    dataloader = LlamaDataLoader(dataset, collate_fn, config.BATCH_SIZE, config.SEED)

    LoraConfig = namedtuple('LoraConfig', ['LORA_R', 'LORA_ALPHA', 'LORA_DROPOUT'])
    loraConfig = LoraConfig(LORA_R=config.LORA_R, LORA_ALPHA=config.LORA_ALPHA, LORA_DROPOUT=config.LORA_DROPOUT)
    with jax.default_device(cpu_device):
        ckpt_path = os.path.join(config.GEMMA_MODEL_PATH, config.MODEL_VERSION)
        params = params_lib.load_and_format_params(ckpt_path)
        model_config = transformer_lib.TransformerConfig.from_params(params, cache_size=config.CACHE_SIZE)

    if config.NUM_SHARDS is None:
        num_gpus = jax.device_count('gpu')
    else:
        num_gpus = config.NUM_SHARDS

    devices = mesh_utils.create_device_mesh((num_gpus,))
    default_mesh = Mesh(devices, axis_names=('p',))

    # bp_mesh = Mesh(devices.reshape((2, 2)), axis_names=('b', 'p',))

    def mesh_sharding(pspec: P, mesh=None) -> NamedSharding:
        if mesh is None:
            mesh = default_mesh
        return NamedSharding(mesh, pspec)

    lora_map = {}

    def param_shard_func(path, v):
        if 'input_embedding' in path:
            return jax.device_put(v, mesh_sharding(P('p', None)))
        elif 'attn_vec_einsum' in path:
            return jax.device_put(v, mesh_sharding(P('p', None, None)))
        elif 'kv_einsum' in path:
            # get the key
            value_shape = v[1].shape # (n_heads, d_m, d_v)
            v_A_shape = (*value_shape[:-1], config.LORA_R) # (n_heads, d_m, r)
            v_B_shape = (value_shape[0], config.LORA_R, *value_shape[-1:]) # (n_heads, r, d_v)
            # create the lora params
            with jax.default_device(cpu_device):
                v_lora_A = jnp.zeros(v_A_shape, dtype=jnp.bfloat16)
                v_lora_B = jnp.zeros(v_B_shape, dtype=jnp.bfloat16)

            lora_map[path] = {
                'v_lora_A': v_lora_A,
                'v_lora_B': v_lora_B
            }

            return jax.device_put(v, mesh_sharding(P(None, None, None, None))) \
                if v.shape[1] == 1 else\
                jax.device_put(v, mesh_sharding(P(None, 'p', None, None)))
        elif 'q_einsum' in path:
            q_A_shape = (*v.shape[:-1], config.LORA_R)
            q_B_shape = (config.LORA_R, *v.shape[-1:])
            # create the lora params
            with jax.default_device(cpu_device):
                q_lora_A = jnp.zeros(q_A_shape, dtype=jnp.bfloat16)
                q_lora_B = jnp.zeros(q_B_shape, dtype=jnp.bfloat16)

            lora_map[path] = {
                'q_lora_A': q_lora_A,
                'q_lora_B': q_lora_B
            }

            return jax.device_put(v, mesh_sharding(P('p', None, None)))
        elif 'gating_einsum' in path:
            return jax.device_put(v, mesh_sharding(P(None, None, 'p')))
        elif 'linear' in path:
            return jax.device_put(v, mesh_sharding(P('p', None)))
        else:
            # replicate across all gpus
            return jax.device_put(v, mesh_sharding(P(*((None,) * len(v.shape)))))

    params = tree.map_structure_with_path(param_shard_func, params)

    # initialize the lora A's to be sampled from a normal distribution

    for k, v in lora_map.items():
        for k1, v1 in v.items():
            if "lora_A" in k1:
                with jax.default_device(cpu_device):
                    key, split = rand.split(key, 2)
                    v[k1] = rand.normal(split, v1.shape, dtype=jnp.bfloat16)

    # shard the lora params
    for k, v in lora_map.items():
        if 'kv_einsum' in k:
            # check if only 1 kv head
            if model_config.num_kv_heads == 1:
                # no sharding
                v['v_lora_A'] = jax.device_put(v['v_lora_A'], mesh_sharding(P(None, None, None)))
                v['v_lora_B'] = jax.device_put(v['v_lora_B'], mesh_sharding(P(None, None, None)))
            else:
                v['v_lora_A'] = jax.device_put(v['v_lora_A'], mesh_sharding(P('p', None, None)))
                v['v_lora_B'] = jax.device_put(v['v_lora_B'], mesh_sharding(P('p', None, None)))
        elif 'q_einsum' in k:
            v['q_lora_A'] = jax.device_put(v['q_lora_A'], mesh_sharding(P('p', None, None)))
            v['q_lora_B'] = jax.device_put(v['q_lora_B'], mesh_sharding(P('p', None, None)))



    if is_process_0 and verbose:
        print('Successfully loaded and sharded model parameters!')

    n_steps = math.ceil(len(dataloader) / config.N_ACCUMULATION_STEPS)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.,
        peak_value=config.LR,
        warmup_steps=n_steps,
        decay_steps=n_steps + 1,
        end_value=config.LR,
    )
    optimizer = optax.adamw(learning_rate=schedule)
    optimizer = optax.MultiSteps(optimizer, config.N_ACCUMULATION_STEPS)
    optimize = optimizer.update

    # opt_state = optimizer.init(params)
    opt_state = optimizer.init(lora_map)

    num_batches = len(dataloader)
    verbosity_freq = num_batches // 100
    jax.clear_caches()

    for epoch in tqdm(range(config.N_EPOCHS)):
        total_loss = jnp.zeros(())

        for step, data_batch in enumerate(dataloader):
            lora_params, opt_state, total_loss, loss, key = train_step_lora(lora_params, loraConfig, params, opt_state,
                                                                            total_loss, data_batch, key)
            if step % verbosity_freq == 0 and verbose:
                print(f'total_loss: {total_loss}, loss: {loss}')

        # save lora params for this epoch
        with open(os.path.join(checkpoint_dir, f'{checkpoint_prefix}_epoch_{epoch}.pickle'), 'wb') as f:
            pickle.dump(lora_params, f)

    with open(os.path.join(checkpoint_dir, f'{checkpoint_prefix}_final.pickle', 'wb')) as f:
        pickle.dump(lora_params, f)



if __name__ == "__main__":

    config = ParagemmaConfig(GEMMA_MODEL_PATH='/home/anique/.cache/kagglehub/models/google/gemma/Flax/2b-it/2',
                             MODEL_VERSION='2b-it')
    dataset_path = Path(__file__).parent.parent.parent / 'alpaca_data_cleaned.json'

    alpaca_dataset = generate_alpaca_dataset(dataset_path, 'train', config, alpaca_mix=0.0)
    train_lora(config, alpaca_dataset, 'checkpoints')


    pass