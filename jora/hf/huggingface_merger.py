import jax

from transformers import LlamaForCausalLM
from torch import nn as tnn
import numpy as np
import jax.numpy as jnp
import torch

from jora.lib.param_utils import load_params
from fire import Fire

def lorize_huggingface_llama(huggingface_path: str, jax_path: str, save_path: str):
    """
    This function takes a huggingface llama model and replaces the q_proj and v_proj weights with the lora merged weights
    :param huggingface_path: path to the huggingface llama model
    :param jax_path: path to the lora merged params
    :param save_path: path to save the updated huggingface llama model
    """
    # load lora merged params in jax format
    with jax.default_device(jax.devices('cpu')[0]):
        jax_params = load_params(jax_path)

    model = LlamaForCausalLM.from_pretrained(huggingface_path)
    print('model loaded')
    # load the huggingface model

    q_projs = []
    for name, weight in model.named_parameters():
        if 'q_proj' in name:
            q_projs.append((name, weight,))

    v_projs = []
    for name, weight in model.named_parameters():
        if 'v_proj' in name:
            v_projs.append((name, weight,))


    jax_qproj = jax_params.model.decoder.attention.q_proj
    # replace the q_proj weights with the lora merged weights
    for i, (name, param) in enumerate(q_projs):
        # get the shape
        shape = param.shape
        # get corresponding jax param
        jax_q_i = jax_qproj[i].reshape(shape[0], -1).T.astype(jnp.float32)
        jax_q_i = torch.from_numpy(np.asarray(jax_q_i))
        param_q_i = tnn.Parameter(jax_q_i)
        model.model.layers[i].self_attn.q_proj.weight = param_q_i

    jax_vproj = jax_params.model.decoder.attention.v_proj
    # replace the v_proj weights with the lora merged weights
    for i, (name, param) in enumerate(v_projs):
        shape = param.shape
        jax_v_i = jax_vproj[i].reshape(shape[0], -1).T.astype(jnp.float32)
        jax_v_i = torch.from_numpy(np.asarray(jax_v_i))
        param_v_i = tnn.Parameter(jax_v_i)
        model.model.layers[i].self_attn.v_proj.weight = param_v_i

    # save the model seperately
    model.save_pretrained(save_path)
    print(f'model saved to {save_path}')

if __name__ == "__main__":
    with torch.no_grad():
        Fire(lorize_huggingface_llama)


