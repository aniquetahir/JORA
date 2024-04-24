# from ast import Not
import jax

from transformers import LlamaForCausalLM, AutoModel
from torch import nn as tnn
import numpy as np
import jax.numpy as jnp
import torch
from gemma import params as params_lib
from jora.lib.param_utils import load_params
from fire import Fire
import pickle
import einops


def lorize_huggingface_gemma(huggingface_path: str, jax_path: str, save_path: str):
    """
    This function takes a huggingface llama model and replaces the q_proj and v_proj weights with the lora merged weights
    :param huggingface_path: path to the huggingface llama model
    :param jax_path: path to the lora merged params
    :param save_path: path to save the updated huggingface llama model
    """
    # load lora merged params in jax format
    with jax.default_device(jax.devices('cpu')[0]):
        with open(jax_path, 'rb') as lora_file:
            jax_params = pickle.load(lora_file)

    model = AutoModel.from_pretrained(huggingface_path)
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

    # transformation einops rearrange 'a b c -> (a c) b'
    for k, v in jax_params.items():
        has_q = False
        has_v = False
        if 'qkv_einsum' in k:
            has_q = True
            has_v = True
        if 'kv_einsum' in k:
            has_v = True
        if 'q_einsum' in k:
            has_q = True

        # get the layer number
        layer_num = int(k[1].split('layer_')[1])
        hf_qproj = model.layers[layer_num].self_attn.q_proj.weight.detach().numpy()
        hf_vproj = model.layers[layer_num].self_attn.v_proj.weight.detach().numpy()

        if has_q:
            q_lora_A = v['q_lora_A']
            q_lora_B = v['q_lora_B']
            merged_q = jnp.einsum('hmr,hrk->hmk', q_lora_A, q_lora_B)
            merged_q = einops.rearrange(merged_q, 'a b c -> (a c) b')
            merged_q = np.array(merged_q)
            merged_q = hf_qproj + merged_q
            
            jax_q_i = torch.from_numpy(np.asarray(merged_q))
            param_q_i = tnn.Parameter(jax_q_i)
            model.layers[layer_num].self_attn.q_proj.weight = param_q_i
            
        if has_v:
            v_lora_A = v['v_lora_A']
            v_lora_B = v['v_lora_B']
            # merge the lora's 
            merged_v = jnp.einsum('hmr,hrk->hmk', v_lora_A, v_lora_B)
            merged_v = einops.rearrange(merged_v, 'a b c -> (a c) b')
            merged_v = np.array(merged_v)
            merged_v = hf_vproj + merged_v

            jax_v_i = torch.from_numpy(np.asarray(merged_v))
            param_v_i = tnn.Parameter(jax_v_i)

            model.layers[layer_num].self_attn.v_proj.weight = param_v_i
        
    # save the model seperately
    model.save_pretrained(save_path)
    print(f'model saved to {save_path}')



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


def lorize_huggingface(huggingface_path: str, jax_path: str, save_path: str, llama2: bool=False, gemma: bool=False):
    if gemma and llama2:
        raise NotImplementedError('model cannot be both llama and gemma')
    elif llama2:
        lorize_huggingface_llama(huggingface_path, jax_path, save_path)
    elif gemma:
        lorize_huggingface_gemma(huggingface_path, jax_path, save_path)
    else:
        raise NotImplementedError('please specify llama2 or gemma, use -h flag for help')


if __name__ == "__main__":
    with torch.no_grad():
        Fire(lorize_huggingface)


