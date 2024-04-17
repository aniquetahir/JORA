import jax
import jax.numpy as jnp
import numpy as np
from lib.param_utils import load_params
import pickle
import tree
import os
# from gemma import params as params_lib
# from gemma import transformer as transformer_lib
# from jora.lib.gemma.common import merge_lora_params
import orbax.checkpoint
import tree


cpu_device = jax.devices('cpu')[0]
gpu_device = jax.devices('gpu')[0]

def merge_q_v(path, value, q_additive, v_additive):
    if 'q_proj' in path:
        return value + q_additive
    elif 'v_proj' in path:
        return value + v_additive
    else:
        return value


def merge_llama2(params_path:str, lora_path:str, output_path:str):
    # load the complete model parameters
    with jax.default_device(cpu_device):
        params = load_params(params_path)

    # load the lora parameters
    with open(lora_path, 'rb') as f:
        lora_params = pickle.load(f)

    print('loaded params (full + lora)')
    print('merging params...')

    # multiply lora components to get final params
    q_lora = jnp.einsum('dmosr,drosn->dmosn', lora_params['q_lora_A'], lora_params['q_lora_B'])
    v_lora = jnp.einsum('dmsr,drsn->dmsn', lora_params['v_lora_A'], lora_params['v_lora_B'])

    # add the final additive params to the original params
    updated_params = tree.map_structure_with_path(lambda p, v: merge_q_v(p, v, q_lora, v_lora), params)

    print('finished merging params')

    with open(output_path, 'wb') as f:
        pickle.dump(updated_params, f)


def merge_gemma(params_path: str, lora_path: str, output_path: str):
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    params_dirs = [x for x in os.listdir(params_path) if os.path.isdir(os.path.join(params_path, x))]
    model_version = params_dirs[0]
    ckpt_path = os.path.join(params_path, model_version)

    orb_params = checkpointer.restore(os.path.abspath(ckpt_path))
    
    with open(lora_path, 'rb') as lora_file:
        lora_params = pickle.load(lora_file)

    # Note: numpy does not support einsum for dtype bfloat16
    for k, v in lora_params.items():
        orb_key = '/'.join(k[:-1])
        if 'kv_einsum' in k:
            breakpoint()
            v_lora_A = v['v_lora_A']
            v_lora_B = v['v_lora_B']
            w = orb_params[orb_key]['w']
            merged_v = w[1] + jnp.einsum('hmr,hrk->hmk', v_lora_A, v_lora_B)
            orb_params[orb_key]['w'] = np.array(jnp.stack([w[0], merged_v]))
        elif 'q_einsum' in k:
            breakpoint()
            w = orb_params[orb_key]['w']
            orb_params[orb_key]['w'] = np.array(w + jnp.einsum('hmr,hrv->hmv', v['q_lora_A'], v['q_lora_B']))
        elif 'qkv_einsum' in k:
            breakpoint()
            w = orb_params[orb_key]['w']
            q_ = w[0]
            k_ = w[1]
            v_ = w[2]
            q_lora_A = v['q_lora_A']
            q_lora_B = v['q_lora_B']
            v_lora_A = v['v_lora_A']
            v_lora_B = v['v_lora_B']
            
            merged_q = q_ + jnp.einsum('hmr,hrk->hmk', q_lora_A, q_lora_B)
            merged_v = v_ + jnp.einsum('hmr,hrk->hmk', v_lora_A, v_lora_B)

            orb_params[orb_key]['w'] = np.array(jnp.stack([merged_q, k_, merged_v]))
    breakpoint()
    print('lora loaded')

    # save the parameters to the desired path
    print("Saving...")
    checkpointer.save(os.path.abspath(output_path), orb_params)
    print("Merged parameters saved successfully!")

def merge_lora(params_path:str, lora_path:str, output_path:str, llama2=False, gemma=False):
    if llama2 and gemma:
        raise ValueError('Please specify only one type of model llama2 or gemma. Use -h for help.')
    if llama2:
        merge_llama2(params_path, lora_path, output_path)
    elif gemma:
        with jax.default_device(cpu_device):
            merge_gemma(params_path, lora_path, output_path)
    else:
        raise ValueError('Please specify type of model llama2 or gemma. Use -h for help.')


