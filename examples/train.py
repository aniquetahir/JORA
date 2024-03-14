from jora import train_lora, ParallamaConfig, AlpacaDataset, generate_alpaca_dataset
from pathlib import Path

def main():
    config = ParallamaConfig(MODEL_SIZE='7B', JAX_PARAMS_PATH='./llama2-7B.pickle',
                             LLAMA2_META_PATH='/media/anique/Data/projects/llama-weights/llama2-7B')
    alpaca_dataset_path = Path(__file__).parent / 'jora' / 'alpaca_data_cleaned.json'
    dataset = generate_alpaca_dataset(alpaca_dataset_path, 'train', config)
    train_lora(config, dataset, 'checkpoints')

if __name__ == '__main__':
    # breakpoint()
    main()
