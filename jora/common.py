from typing import NamedTuple
import os
import sys
from os.path import join as pjoin
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
from typing import Any, Callable, Optional

import sys
import jora.lib
sys.modules['lib'] = jora.lib

from .lib.dataloader import LlamaDataLoader
from .lib.alpaca_data import AlpacaDataset, TrainData, alpaca_collate_fn_train
from .lib.loss import cross_entropy_loss
from .lib.model import Llama, llama_model, model_config_llama2_7B,llama_model_lora, model_config_llama2_13B, model_config_llama2_70B, ModelConfig
from .lib.param_utils import load_params, save_params
from .hf.hf_to_jax import hf_to_jax
from os.path import join as pjoin

from jax.sharding import Mesh, NamedSharding, PartitionSpec, PositionalSharding
from functools import partial
from collections import namedtuple
import pickle


optimize: Optional[Callable]
active_model_config: ModelConfig

model_config_mapping = {
    '7B': model_config_llama2_7B,
    '13B': model_config_llama2_13B,
    '70B': model_config_llama2_70B
}

cpu_devices = jax.devices('cpu')
cpu_sharding = NamedSharding(Mesh(cpu_devices, ('D',)), PartitionSpec(None))

gpu_devices = jax.devices('gpu')
gpu_sharding_mp = PositionalSharding(gpu_devices)
gpu_sharding_mp = gpu_sharding_mp.reshape((1, len(gpu_devices)))


class ParallamaConfig(NamedTuple):
    JAX_PARAMS_PATH: str
    LLAMA2_META_PATH: str # e.g. '/tmp/llama2-13B'
    MODEL_SIZE: str # '7B', '13B', '70B'
    NUM_GPUS: int = None
    LORA_R: int = 16
    LORA_ALPHA: int = 16
    LORA_DROPOUT: float = 0.05
    LR: float = 0.0001
    BATCH_SIZE: int = 1
    N_ACCUMULATION_STEPS: int = 8
    MAX_SEQ_LEN = 2000
    N_EPOCHS: int = 7
    SEED: int = 420


is_process_0 = jax.process_index() == 0
cpu_device = jax.devices('cpu')[0]
gpu_device = jax.devices('gpu')[0]

@jax.value_and_grad
def train_forward_lora(lora_params, lora_config, params: Llama, data_batch: TrainData, *, key: Array):
    seq, seq_mask, labels, labels_mask = data_batch
    outputs = llama_model_lora(lora_params, lora_config, params.model, seq, seq_mask, key=key, model_config=active_model_config)
    logits = outputs @ params.lm_head
    loss = cross_entropy_loss(logits, labels, mask=labels_mask)
    return loss

@partial(jax.jit, static_argnames=('lora_config',))
def train_step_lora(lora_params, lora_config, params: Llama, opt_state: Any, total_loss: Array, data_batch: TrainData, key: Array) -> tuple[Llama, Any, Array, Array, Array]:
    key, subkey = rand.split(key)
    loss, grads = train_forward_lora(lora_params, lora_config, params, data_batch, key=subkey)
    total_loss += loss
    updates, opt_state = optimize(grads, opt_state, lora_params)  # type: ignore
    lora_params = optax.apply_updates(lora_params, updates)
    return lora_params, opt_state, total_loss, loss, key


def generate_alpaca_dataset(path: str, split: str, config: ParallamaConfig, split_percentage=0.8, alpaca_mix=0.3):
    """
    Generate an AlpacaDataset object
    :param path:
    :param split: 'train' or 'test'
    :param config:
    :param split_percentage:
    :param alpaca_mix:
    :return:
    """
    tokenizer = LlamaTokenizer.from_pretrained(config.LLAMA2_META_PATH)
    dataset = AlpacaDataset(split=split, path=path, split_percentage=split_percentage, tokenizer=tokenizer, alpaca_mix=alpaca_mix)
    return dataset

def train_lora(config: ParallamaConfig, train_dataset: AlpacaDataset, checkpoint_dir: str, checkpoint_prefix='jax_lora', verbose=True) -> None:
    global optimize
    global active_model_config

    if not config.JAX_PARAMS_PATH or not config.LLAMA2_META_PATH:
        raise ValueError('Please provide JAX_PARAMS_PATH and LLAMA2_HF_PATH in the config')
    
    if not config.MODEL_SIZE or config.MODEL_SIZE not in ['7B', '13B', '70B']:
        raise ValueError('Please provide a valid MODEL_SIZE in the config (7B, 13B, 70B)')

    # check if file exists at JAX_PARAMS_PATH
    if not os.path.exists(config.JAX_PARAMS_PATH):
        # call the function for creating the jax version of huggingface model
        print('Converting HuggingFace model to JAX...')
        hf_to_jax(config.MODEL_SIZE, config.LLAMA2_META_PATH, config.JAX_PARAMS_PATH)
        print('Conversion complete!')

    # IMPORTANT: these values should be set before jit compilation occurs. Otherwise, it will compile with the wrong values
    active_model_config = model_config_mapping[config.MODEL_SIZE]

    # check if the checkpoint directory exists and create it if it doesn't
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    key = rand.PRNGKey(config.SEED)
    tokenizer = LlamaTokenizer.from_pretrained(config.LLAMA2_META_PATH)
    # dataset = AlpacaDataset(split='train', path='./merged_dataset_insta_4chan.json', tokenizer=tokenizer, max_len=max_len, alpaca_mix=0.)
    dataset = train_dataset
    collate_fn = partial(alpaca_collate_fn_train, tokenizer, config.MAX_SEQ_LEN)

    dataloader = LlamaDataLoader(dataset, collate_fn, config.BATCH_SIZE, config.SEED)

    LoraConfig = namedtuple('LoraConfig', ['LORA_R', 'LORA_ALPHA', 'LORA_DROPOUT'])
    loraConfig = LoraConfig(LORA_R=config.LORA_R, LORA_ALPHA=config.LORA_ALPHA, LORA_DROPOUT=config.LORA_DROPOUT)
    with jax.default_device(cpu_device):
        params = load_params(config.JAX_PARAMS_PATH)

    if config.NUM_GPUS is None:
        num_gpus = jax.device_count('gpu')
    else:
        num_gpus = config.NUM_GPUS

    devices = mesh_utils.create_device_mesh((num_gpus, ))
    default_mesh = Mesh(devices, axis_names=('p',))
    # bp_mesh = Mesh(devices.reshape((2, 2)), axis_names=('b', 'p',))

    def mesh_sharding(pspec: P, mesh=None) -> NamedSharding:
        if mesh is None:
            mesh = default_mesh
        return NamedSharding(mesh, pspec)

    def param_shard_func(path, v):
        # breakpoint()
        if 'q_proj' in path:
            return jax.device_put(v, mesh_sharding(P(None, None, None, 'p', None)))
            # parallelize on the 3rd index (count 5)
        elif 'k_proj' in path or 'v_proj' in path:
            return jax.device_put(v, mesh_sharding(P(None, None, 'p', None)))
            # parallelize on the 2nd index (count 4)
        elif 'out_proj' in path:
            # parallelize on the 2nd index (count 5)
            return jax.device_put(v, mesh_sharding(P(None, None, 'p', None, None)))
        elif 'embedding' in path or 'post_attn_norm' in path or 'input_norm' in path:
            # parallelize on the 1st index (count 2)
            return jax.device_put(v, mesh_sharding(P(None, 'p')))
        elif 'gate_proj' in path or 'up_proj' in path:
            # parallelize on the 1st index (count 3)
            return jax.device_put(v, mesh_sharding(P(None, 'p', None)))
        elif 'down_proj' in path:
            # parallelize on the 0th index (count 3)
            return jax.device_put(v, mesh_sharding(P('p', None, None)))
        else:
            # replicate across all gpus
            return jax.device_put(v, mesh_sharding(P(*((None,) * len(v.shape)))))


    params = tree.map_structure_with_path(param_shard_func, params)

    # extract q_proj and v_proj from params
    q_proj_shape, v_proj_shape = params.model.decoder.attention.q_proj.shape, params.model.decoder.attention.v_proj.shape

    # create lora params from q_proj and v_proj
    DB, H, N_REP, N_HEADS, D_K = q_proj_shape
    assert v_proj_shape == (DB, H, N_HEADS, D_K,)

    # create lora params from q_proj and v_proj
    key, split_qa, split_va = rand.split(key, 3)
    with jax.default_device(cpu_device):
        q_lora_A = rand.normal(split_qa, (DB, H, N_REP, N_HEADS, config.LORA_R), dtype=jnp.bfloat16)
        q_lora_B = jnp.zeros((DB, config.LORA_R, N_REP, N_HEADS, D_K), dtype=jnp.bfloat16)
        v_lora_A = rand.normal(split_va, (DB, H, N_HEADS, config.LORA_R), dtype=jnp.bfloat16)
        v_lora_B = jnp.zeros((DB, config.LORA_R, N_HEADS, D_K), dtype=jnp.bfloat16)

    # breakpoint()
    # shard the lora params
    q_lora_A = jax.device_put(q_lora_A, mesh_sharding(P(None, None, None, 'p', None)))
    q_lora_B = jax.device_put(q_lora_B, mesh_sharding(P(None, None, None, 'p', None)))
    v_lora_A = jax.device_put(v_lora_A, mesh_sharding(P(None, None, 'p', None)))
    v_lora_B = jax.device_put(v_lora_B, mesh_sharding(P(None, None, 'p', None)))

    lora_params = {
        'q_lora_A': q_lora_A,
        'q_lora_B': q_lora_B,
        'v_lora_A': v_lora_A,
        'v_lora_B': v_lora_B,
    }

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
    opt_state = optimizer.init(lora_params)

    num_batches = len(dataloader)
    verbosity_freq = num_batches // 100
    jax.clear_caches()



    for epoch in tqdm(range(config.N_EPOCHS)):
        total_loss = jnp.zeros(())

        for step, data_batch in enumerate(dataloader):
            lora_params, opt_state, total_loss, loss, key = train_step_lora(lora_params, loraConfig, params, opt_state, total_loss, data_batch, key)
            if step % verbosity_freq == 0 and verbose:
                print(f'total_loss: {total_loss}, loss: {loss}')

        # save lora params for this epoch
        with open(os.path.join(checkpoint_dir, f'{checkpoint_prefix}_epoch_{epoch}.pickle'), 'wb') as f:
            pickle.dump(lora_params, f)

    with open(os.path.join(checkpoint_dir, f'{checkpoint_prefix}_final.pickle', 'wb')) as f:
        pickle.dump(lora_params, f)
