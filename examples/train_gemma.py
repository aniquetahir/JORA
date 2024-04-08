from jora import ParagemmaConfig, generate_alpaca_dataset_gemma, train_lora_gemma
from pathlib import Path

def main():
    config = ParagemmaConfig(GEMMA_MODEL_PATH='/home/anique/.cache/kagglehub/models/google/gemma/Flax/2b-it/2',
                             MODEL_VERSION='2b-it')
    dataset_path = Path(__file__).parent.parent / 'jora' / 'alpaca_data_cleaned.json'
    alpaca_dataset = generate_alpaca_dataset_gemma(dataset_path, 'train', config, alpaca_mix=0.)
    train_lora_gemma(config, alpaca_dataset, 'checkpoints')

if __name__ == '__main__':
    main()