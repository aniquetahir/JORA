import sentencepiece as spm
import jax
import jax.numpy as jnp
from collections import namedtuple
from gemma import transformer as transformer_lib


class GemmaTokenizer:
    def __init__(self,
                 spm_processor: spm.SentencePieceProcessor):
        self._spm_processor = spm_processor

    @property
    def pad_id(self) -> int:
        """Fast access to the pad id."""
        return self._spm_processor.pad_id()
    @property
    def bos_token_id(self) -> int:
        """Fast access to the bos token id."""
        return self._spm_processor.bos_id()

    @property
    def eos_token_id(self) -> int:
        """Fast access to the eos token id."""
        return self._spm_processor.eos_id()


    def tokenize(self,
                 example: str | bytes,
                 prefix: str = '',
                 suffix: str = '',
                 add_eos: bool = True) -> jax.Array:
        """
        Tokenization function.

        Args:
          example: input string to tokenize.
          prefix:  prefix to add to the input string.
          suffix:  suffix to add to the input string.
          add_eos: if True, add an end of sentence token at the end of the output
                   sequence.
        Returns:
          Tokens corresponding to the input string.
        """
        int_list = [self._spm_processor.bos_id()]
        int_list.extend(self._spm_processor.EncodeAsIds(prefix + example + suffix))
        if add_eos:
            int_list.append(self._spm_processor.eos_id())

        return int_list

    def __call__(self, input, return_attention_mask: bool = False, add_special_tokens: bool = True, **kwargs):
        if return_attention_mask:
            raise NotImplementedError("return_attention_mask is not implemented for GemmaTokenizer")
        inp_ids = self.tokenize(input, add_eos=add_special_tokens)
        # remove bos if add_special_tokens is False
        if not add_special_tokens:
            inp_ids = inp_ids[1:]
        return namedtuple('GemmaTokenizerOutput',
                          ['input_ids'])(inp_ids)

    def to_string(self, tokens: jax.Array) -> str:
        """Convert an array of tokens to a string."""
        return self._spm_processor.EncodeIds(tokens.tolist())


def get_attention_mask_and_positions(example: jax.Array,
                                     pad_id: int,
                                     ) -> tuple[jax.Array, jax.Array]:
    """Builds the position and attention mask vectors from the given tokens."""
    pad_mask = example != pad_id
    current_token_position = transformer_lib.build_positions_from_mask(pad_mask)
    attention_mask = transformer_lib.make_causal_attn_mask(pad_mask)
    return current_token_position, attention_mask
