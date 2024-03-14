from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

import fire
import jax
import jax.numpy as jnp
from transformers import LlamaForCausalLM

from jora.lib.model import check_llama, model_config_llama1_7B, model_config_llama2_70B, model_config_llama2_7B, model_config_llama2_13B
from jora.lib.param_utils import convert_llama, save_params
from os.path import join as pjoin

pairs = {
    '7B':  model_config_llama2_7B,
    '13B': model_config_llama2_13B,
    '70B': model_config_llama2_70B
}


def hf_to_jax(model_type:str, hf_path: str, jax_path: str):
    """
    Convert a HuggingFace model to JAX.
    :param model_type: The model type to convert.
    :param hf_path: The path to the HuggingFace model.
    :param jax_path: The path to save the JAX model.
    """
    model_config = pairs[model_type]
    model_pt = LlamaForCausalLM.from_pretrained(hf_path)
    with jax.default_device(jax.devices('cpu')[0]):
        params = convert_llama(model_pt, model_config=model_config)
        params = jax.tree_map(lambda x: x.astype(jnp.bfloat16), params)
    check_llama(params, model_config=model_config)
    save_params(params, jax_path)

if __name__ == '__main__':
    fire.Fire(hf_to_jax)