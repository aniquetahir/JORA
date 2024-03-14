from .huggingface_merger import lorize_huggingface_llama
from fire import Fire

if __name__ == "__main__":
    Fire(lorize_huggingface_llama)