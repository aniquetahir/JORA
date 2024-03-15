import os
import sys
from os.path import join as pjoin
import tree
# os.chdir('/scratch/artahir/compressed_code/llama2jax/llama-2-jax')
# sys.path.append('/scratch/artahir/compressed_code/llama2jax/llama-2-jax')

import jax
from jax import Array
from jax.experimental.multihost_utils import process_allgather
import jax.numpy as jnp
import jax.random as rand

# Define the mesh
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils

from jax_smi import initialise_tracking
initialise_tracking()

import math
import optax
import time
from transformers import LlamaTokenizer
from tqdm import tqdm
from typing import Any, Callable, Optional

from jora.lib.dataloader import LlamaDataLoader
from jora.lib.alpaca_data import AlpacaDataset, TrainData, alpaca_collate_fn_train
from jora.lib.loss import cross_entropy_loss
from jora.lib.model import Llama, llama_model, model_config_llama2_7B,llama_model_lora
from jora.lib.param_utils import load_params, save_params
from os.path import join as pjoin

from jax.sharding import Mesh, NamedSharding, PartitionSpec, PositionalSharding
from functools import partial
from collections import namedtuple
import pickle

optimize: Optional[Callable]
# BASE_WEIGHTS_PATH = '/media/anique/Data/projects/llama-weights'
JAX_PARAMS_PATH = '../../../llama/jax_weights/merged.pickle'
LLAMA2_META_PATH = '../../../llama/hf_weights'
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

cpu_devices = jax.devices('cpu')
cpu_sharding = NamedSharding(Mesh(cpu_devices, ('D',)), PartitionSpec(None))

gpu_devices = jax.devices('gpu')
gpu_sharding_mp = PositionalSharding(gpu_devices)
gpu_sharding_mp = gpu_sharding_mp.reshape((1, len(gpu_devices)))


@jax.value_and_grad
def train_forward_lora(lora_params, lora_config, params: Llama, data_batch: TrainData, *, key: rand.KeyArray):
    seq, seq_mask, labels, labels_mask = data_batch
    outputs = llama_model_lora(lora_params, lora_config, params.model, seq, seq_mask, key=key, model_config=model_config_llama2_7B)
    logits = outputs @ params.lm_head
    loss = cross_entropy_loss(logits, labels, mask=labels_mask)
    return loss

@partial(jax.jit, static_argnames=('lora_config',))
def train_step_lora(lora_params, lora_config, params: Llama, opt_state: Any, total_loss: Array, data_batch: TrainData, key: rand.KeyArray) -> tuple[Llama, Any, Array, Array, rand.KeyArray]:
    key, subkey = rand.split(key)
    loss, grads = train_forward_lora(lora_params, lora_config, params, data_batch, key=subkey)
    total_loss += loss
    updates, opt_state = optimize(grads, opt_state, lora_params)  # type: ignore
    lora_params = optax.apply_updates(lora_params, updates)
    return lora_params, opt_state, total_loss, loss, key

lr = 0.0001
batch_size = 1
n_accumulation_steps = 8
max_len = 512
n_epochs = 7
seed = 3407

# initialise_tpu('v4-16', n_devices=8, rank=0)
is_process_0 = jax.process_index() == 0
cpu_device = jax.devices('cpu')[0]
gpu_device = jax.devices('gpu')[0]


def main(num_gpus=4) -> None:
    global optimize

    key = rand.PRNGKey(seed)
    tokenizer = LlamaTokenizer.from_pretrained(pjoin(LLAMA2_META_PATH, 'llama2-7B'))
    dataset = AlpacaDataset(split='train', path='./alpaca_data_cleaned.json', tokenizer=tokenizer, max_len=max_len, alpaca_mix=0.)
    collate_fn = partial(alpaca_collate_fn_train, tokenizer, max_len)

    dataloader = LlamaDataLoader(dataset, collate_fn, batch_size, seed)

    LoraConfig = namedtuple('LoraConfig', ['LORA_R', 'LORA_ALPHA', 'LORA_DROPOUT'])
    loraConfig = LoraConfig(LORA_R=LORA_R, LORA_ALPHA=LORA_ALPHA, LORA_DROPOUT=LORA_DROPOUT)
    with jax.default_device(cpu_device):
        params = load_params(JAX_PARAMS_PATH)

    devices = mesh_utils.create_device_mesh((num_gpus, ))
    default_mesh = Mesh(devices, axis_names=('p',))
    # bp_mesh = Mesh(devices.reshape((2,2)), axis_names=('b', 'p',))

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
        q_lora_A = rand.normal(split_qa, (DB, H, N_REP, N_HEADS, LORA_R), dtype=jnp.bfloat16)
        q_lora_B = jnp.zeros((DB, LORA_R, N_REP, N_HEADS, D_K), dtype=jnp.bfloat16)
        v_lora_A = rand.normal(split_va, (DB, H, N_HEADS, LORA_R), dtype=jnp.bfloat16)
        v_lora_B = jnp.zeros((DB, LORA_R, N_HEADS, D_K), dtype=jnp.bfloat16)

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

    if is_process_0:
        print('Successfully loaded and sharded model parameters!')

    n_steps = math.ceil(len(dataloader) / n_accumulation_steps)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.,
        peak_value=lr,
        warmup_steps=n_steps,
        decay_steps=n_steps + 1,
        end_value=lr,
    )
    optimizer = optax.adamw(learning_rate=schedule)
    optimizer = optax.MultiSteps(optimizer, n_accumulation_steps)
    optimize = optimizer.update

    # opt_state = optimizer.init(params)
    opt_state = optimizer.init(lora_params)

    num_batches = len(dataloader)
    verbosity_freq = num_batches // 100
    jax.clear_caches()

    for epoch in tqdm(range(n_epochs)):
        step_loss = 0.0
        total_loss = jnp.zeros(())

        for step, data_batch in enumerate(dataloader):
            start_time = time.time()
            lora_params, opt_state, total_loss, loss, key = train_step_lora(lora_params, loraConfig, params, opt_state, total_loss, data_batch, key)
            if step % verbosity_freq == 0:
                print(f'total_loss: {total_loss}, loss: {loss}')


    # gathered_params = process_allgather(lora_params)
    # if is_process_0:
    #     save_params(gathered_params, f'{lora_final_gathered}.pickle')  # type: ignore


if __name__ == "__main__":
    main()


