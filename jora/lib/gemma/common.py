from typing import NamedTuple
import os
from collections import defaultdict
import sys
from os.path import join as pjoin
from pathlib import Path
import tree
from pprint import pprint

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
from .gemma_utils import GemmaTokenizer, get_attention_mask_and_positions
from .gemma_config import GemmaConfig, GemmaConfig2B, GemmaConfig7B, GEMMA_VERSIONS
from gemma import params as params_lib
from gemma import sampler as sampler_lib
from gemma import transformer as transformer_lib
import sentencepiece as spm

from rich.progress import Progress

optimize: Optional[Callable]
active_model_config: GemmaConfig

model_config_mapping = {
    '2b': GemmaConfig2B,
    '2b-it': GemmaConfig2B,
    '7b': GemmaConfig7B,
    '7b-it': GemmaConfig7B,
    '1.1-2b-it': GemmaConfig2B,
    '1.1-7b-it': GemmaConfig7B
}

cpu_devices = jax.devices('cpu')
cpu_sharding = NamedSharding(Mesh(cpu_devices, ('D',)), PartitionSpec(None))

gpu_devices = jax.devices('gpu')
gpu_sharding_mp = PositionalSharding(gpu_devices)
gpu_sharding_mp = gpu_sharding_mp.reshape((1, len(gpu_devices)))


class ParagemmaConfig(NamedTuple):
    GEMMA_MODEL_PATH: str  # e.g. '/tmp/llama2-13B'
    MODEL_VERSION: str  # '2b', '7b', '2b-it', '7b-it'
    NUM_SHARDS: int
    LORA_R: int
    LORA_ALPHA: int
    LORA_DROPOUT: float
    LR: float
    BATCH_SIZE: int
    N_ACCUMULATION_STEPS: int
    MAX_SEQ_LEN: int
    N_EPOCHS: int
    SEED: int
    CACHE_SIZE: int  # Numbber of steps in the transformer's cache

ParagemmaConfig.__new__.__defaults__ = (
    None,
    None,
    None,
    16,
    16,
    0.05,
    0.0001,
    1,
    8,
    2000,
    7,
    420,
    30
)


is_process_0 = jax.process_index() == 0
cpu_device = jax.devices('cpu')[0]
gpu_device = jax.devices('gpu')[0]


def forward_and_loss_fn(lora_params,
                        pretrained_params,
                        *,
                        model: transformer_lib.Transformer,
                        input_tokens: jax.Array,  # Shape [B, L]
                        input_mask: jax.Array,  # Shape [B, L]
                        positions: jax.Array,  # Shape [B, L]
                        attention_mask: jax.Array,  # [B, L, L]
                        ) -> jax.Array:
    """Foward pass and loss function.

    Args:
      params: model's input parameters.
      model: gemma transformer model to call.
      input_tokens: input tokens sequence, shape [B, L].
      input_mask: tokens to ignore when computing the loss, shape [B, L].
      positions: relative position of each token, shape [B, L].
      attention_mask: input attention mask, shape [B, L].

    Returns:
      Softmax cross-entropy loss for the next-token prediction task.
    """
    params = merge_lora_params(pretrained_params, lora_params)
    # Foward pass on the input data.
    # No attention cache is needed here.
    logits, _ = model.apply(
        {'params': params['transformer']},
        input_tokens,
        positions,
        None,  # Attention cache is None.
        attention_mask,
    )

    # Exclude the last step as it does not appear in the targets.
    logits = logits[0, :-1]

    # Similarly, the first token cannot be predicteds.
    target_tokens = input_tokens[0, 1:]
    target_mask = input_mask[0, 1:]

    # Convert the target labels into one-hot encoded vectors.
    one_hot = jax.nn.one_hot(target_tokens, logits.shape[-1])

    # Don't update on unwanted tokens.
    one_hot = one_hot * target_mask.astype(one_hot.dtype)[..., None]

    # Normalisation factor.
    norm_factor = 1 / (jnp.sum(target_mask) + 1e-8)

    # Return the nll loss.
    return -jnp.sum(jax.nn.log_softmax(logits) * one_hot) * norm_factor


def train_step_lora(model: transformer_lib.Transformer,
               lora_params,
               params,
               opt_state: optax.OptState,
               total_loss: jax.Array,
               input_tokens: jax.Array,
               target_mask: jax.Array,
               positions: jax.Array,
               attention_mask: jax.Array,
               ) -> tuple[jax.Array, optax.OptState]:
    """Train step.

    Args:
      model: gemma transformer model.
      params: model's input parameters.
      optimizer: optax optimizer to use.
      opt_state: input optimizer's state.
      pad_id: id of the pad token.
      example: input batch.

    Returns:
      Training loss, updated parameters, updated optimizer state.
    """
    # TODO Lora dropout

    # Forward and backward passes
    train_loss, grads = jax.value_and_grad(forward_and_loss_fn)(
                                                                lora_params,
                                                                params,
                                                                model=model,
                                                                input_tokens=input_tokens,
                                                                input_mask=target_mask,
                                                                positions=positions,
                                                                attention_mask=attention_mask)
    total_loss += train_loss
    # Update the parameters
    updates, opt_state = optimize(grads, opt_state, lora_params)
    lora_params = optax.apply_updates(lora_params, updates)
    return lora_params, opt_state, total_loss, train_loss


def validation_step(model: transformer_lib.Transformer,
                    params,
                    pad_id: int,
                    example,
                    ):
    # TODO lorize the step
    positions, attention_mask = get_attention_mask_and_positions(example.input_tokens, pad_id)
    val_loss = forward_and_loss_fn(params,
                                   model=model,
                                   input_tokens=example.input_tokens,
                                   input_mask=example.target_mask,
                                   positions=positions,
                                   attention_mask=attention_mask)
    return val_loss


def merge_lora_params(params, lora_params: Dict):
    def merge_fn(path, v):
        # h - num heads, m - model dim, r - lora dim, k - key dim, v - value dim
        if 'kv_einsum' in path:
            v_lora_A = lora_params[path]['v_lora_A']
            v_lora_B = lora_params[path]['v_lora_B']
            merged_V = v[1] + jnp.einsum('hmr,hrk->hmk', v_lora_A, v_lora_B)  # this gives us the same dimension as v[1]
            return jnp.stack([v[0], merged_V])
        elif 'q_einsum' in path:
            return v + jnp.einsum('hmr,hrv->hmv', lora_params[path]['q_lora_A'], lora_params[path]['q_lora_B'])
        elif 'qkv_einsum' in path:
            q_ = v[0]
            k_ = v[1]
            v_ = v[2]
            q_lora_A = lora_params[path]['q_lora_A']
            q_lora_B = lora_params[path]['q_lora_B']
            v_lora_A = lora_params[path]['v_lora_A']
            v_lora_B = lora_params[path]['v_lora_B']
            
            merged_q = q_ + jnp.einsum('hmr,hrk->hmk', q_lora_A, q_lora_B)
            merged_v = v_ + jnp.einsum('hmr,hrk->hmk', v_lora_A, v_lora_B)

            return jnp.stack([merged_q, k_, merged_v])
        else:
            return v

    merged_params = tree.map_structure_with_path(merge_fn, params)
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
    dataset = AlpacaDataset(split=split, path=path, split_percentage=split_percentage, tokenizer=tokenizer,
                            alpaca_mix=alpaca_mix)
    return dataset


def train_lora(config: ParagemmaConfig, train_dataset: AlpacaDataset, checkpoint_dir: str, checkpoint_prefix='jax_lora',
               verbose=True) -> None:
    global optimize
    global active_model_config

    if not config.GEMMA_MODEL_PATH or not config.MODEL_VERSION:
        raise ValueError(
            'Please provide a valid GEMMA_MODEL_PATH and MODEL_VERSION in the config (2b, 7b, 2b-it, 7b-it)')

    if not config.MODEL_VERSION in GEMMA_VERSIONS:
        raise ValueError(f'Please provide a valid MODEL_SIZE in the config ({" ,".join(GEMMA_VERSIONS)}) \nFor Gemma 1.1 use 2b-it or 7b-it')

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
    collate_fn_train = train_dataset.get_collate_fn_train()
    collate_fn = partial(collate_fn_train, tokenizer, config.MAX_SEQ_LEN)

    dataloader = LlamaDataLoader(dataset, collate_fn, config.BATCH_SIZE, config.SEED)

    # LoraConfig = namedtuple('LoraConfig', ['LORA_R', 'LORA_ALPHA', 'LORA_DROPOUT'])
    # loraConfig = LoraConfig(LORA_R=config.LORA_R, LORA_ALPHA=config.LORA_ALPHA, LORA_DROPOUT=config.LORA_DROPOUT)
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
            value_shape = v[1].shape  # (n_heads, d_m, d_v)
            v_A_shape = (*value_shape[:-1], config.LORA_R)  # (n_heads, d_m, r)
            v_B_shape = (value_shape[0], config.LORA_R, value_shape[-1])  # (n_heads, r, d_v)
            # create the lora params
            with jax.default_device(cpu_device):
                v_lora_A = jnp.zeros(v_A_shape, dtype=jnp.bfloat16)
                v_lora_B = jnp.zeros(v_B_shape, dtype=jnp.bfloat16)

            lora_map[path] = {
                'v_lora_A': v_lora_A,
                'v_lora_B': v_lora_B
            }

            return jax.device_put(v, mesh_sharding(P(None, None, None, None))) \
                if v.shape[1] == 1 else \
                jax.device_put(v, mesh_sharding(P(None, 'p', None, None)))
        elif 'q_einsum' in path:
            q_A_shape = (*v.shape[:-1], config.LORA_R)  # (n_heads, d_m, r)
            q_B_shape = (v.shape[0], config.LORA_R, v.shape[-1])  # (n_heads, r, d_k)
            # create the lora params
            with jax.default_device(cpu_device):
                q_lora_A = jnp.zeros(q_A_shape, dtype=jnp.bfloat16)
                q_lora_B = jnp.zeros(q_B_shape, dtype=jnp.bfloat16)

            lora_map[path] = {
                'q_lora_A': q_lora_A,
                'q_lora_B': q_lora_B
            }

            return jax.device_put(v, mesh_sharding(P('p', None, None)))
        elif 'qkv_einsum' in path:
            value_shape = v[1].shape
            v_A_shape = (*value_shape[:-1], config.LORA_R)
            v_B_shape = (value_shape[0], config.LORA_R)

            q_A_shape, q_B_shape = v_A_shape, v_B_shape

            with jax.default_device(cpu_device):
                v_lora_A = jnp.zeros(v_A_shape, dtype=jnp.bfloat16)
                v_lora_B = jnp.zeros(v_B_shape, dtype=jnp.bfloat16)
                q_lora_A = jnp.zeros(q_A_shape, dtype=jnp.bfloat16) 
                q_lora_B = jnp.zeros(q_B_shape, dtype=jnp.bfloat16) 

            lora_map[path] = {
                'v_lora_A': v_lora_A, 
                'v_lora_B': v_lora_B, 
                'q_lora_A': q_lora_A,
                'q_lora_B': q_lora_B
            }
            
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

    # merge_lora_params(params, lora_map)

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

    # combined_params = merge_lora_params(params, lora_map)
    # compare the shapes
    # shape_equality = jax.tree_map(lambda x, y: x.shape == y.shape, params, combined_params)
    # pprint(shape_equality)


    compiled_train_step = jax.jit(train_step_lora, static_argnames=('model',))
    model = transformer_lib.Transformer(model_config)

    progress = Progress()
    p_task_epoch = progress.add_task("[red]Epochs...", total=config.N_EPOCHS)
    p_task_step = progress.add_task("[green]Batches...", total=len(dataloader))
    progress.start()

    for epoch in range(config.N_EPOCHS):
        progress.update(p_task_epoch, advance=1)
        total_loss = jnp.zeros(())

        for step, data_batch in enumerate(dataloader):
            # input_tokens: input tokens sequence, shape[B, L].
            input_tokens = data_batch.seq
            # input_mask: tokens to ignore when computing the loss, shape [B, L].
            input_mask = data_batch.seq_mask != data_batch.labels_mask
            positions, att_mask = get_attention_mask_and_positions(input_tokens, tokenizer.pad_id)

            lora_map, opt_state, total_loss, loss = compiled_train_step(
                model,
                lora_map, params, opt_state, total_loss,
                input_tokens, input_mask, positions, att_mask)
            if step % verbosity_freq == 0 and verbose:
                progress.update(p_task_step, description=f'[green]Batches...\n total_loss: {total_loss}, loss: {loss}')
                # print(f'total_loss: {total_loss}, loss: {loss}')
            progress.update(p_task_step, advance=1)

        progress.update(p_task_epoch, description=f'[red]Epochs...\n total_loss: {total_loss}, loss: {loss}')
        # reset the inner loop progress bar
        progress.reset(p_task_step)


        # save lora params for this epoch
        with open(os.path.join(checkpoint_dir, f'{checkpoint_prefix}_epoch_{epoch}.pickle'), 'wb') as f:
            pickle.dump(lora_map, f)

    progress.stop()
    with open(os.path.join(checkpoint_dir, f'{checkpoint_prefix}_final.pickle'), 'wb') as f:
        pickle.dump(lora_map, f)


if __name__ == "__main__":
    config = ParagemmaConfig(GEMMA_MODEL_PATH='/home/anique/.cache/kagglehub/models/google/gemma/Flax/2b-it/2',
                             MODEL_VERSION='2b-it')
    dataset_path = Path(__file__).parent.parent.parent / 'alpaca_data_cleaned.json'

    alpaca_dataset = generate_alpaca_dataset(dataset_path, 'train', config, alpaca_mix=0.0)
    train_lora(config, alpaca_dataset, 'checkpoints')

    pass
