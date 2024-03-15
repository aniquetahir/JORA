import json
from jora.lib.alpaca_data import AlpacaDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup, default_data_collator
import datasets
import torch
from accelerate import Accelerator
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
import deepspeed
import gc, psutil, threading
from tqdm import tqdm

LR = 0.0001
EPOCHS = 7
BATCH_SIZE = 1
n_accumulation_steps = 8

def b2mb(x):
    return int(x / 2**20)

class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")

def preprocess_function(examples, tokenizer=None, max_length=512):
    batch_size = len(examples['prompt'])
    model_inputs = tokenizer(examples['prompt'])
    labels = tokenizer(examples['response'], add_special_tokens=False)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.eos_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
    model_inputs["labels"] = labels["input_ids"]

    model_inputs['input_ids'] = torch.stack(model_inputs['input_ids'])
    model_inputs['attention_mask'] = torch.stack(model_inputs['attention_mask'])
    model_inputs['labels'] = torch.stack(model_inputs['labels'])
    return model_inputs

accelerator = Accelerator()
LLAMA_WEIGHTS_PATH = '/media/anique/Data/projects/llama-weights/llama2-7B'


if __name__ == "__main__":
    # Load the alpaca dataset
    with open('alpaca_data_cleaned.json', 'r') as f:
        alpaca_data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(LLAMA_WEIGHTS_PATH)
    dataset = AlpacaDataset(path='alpaca_data_cleaned.json', split='train', tokenizer=tokenizer, split_percentage=0.8, alpaca_mix=0.0)


    list_dataset = []
    for sample in dataset:
        list_dataset.append(
            {
                'prompt': sample[0],
                'response': sample[1]
            }
        )

    hf_dataset = datasets.Dataset.from_list(list_dataset)
    hf_dataset = hf_dataset.train_test_split(test_size=0.2)
    hf_dataset

    with accelerator.main_process_first():
        hf_dataset_tk = hf_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer, 'max_length': 512}, remove_columns=hf_dataset['train'].column_names)
    accelerator.wait_for_everyone()

    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.05)
    train_dataloader = DataLoader(hf_dataset_tk['train'], batch_size=BATCH_SIZE, shuffle=True, collate_fn=default_data_collator)

    model = AutoModelForCausalLM.from_pretrained(LLAMA_WEIGHTS_PATH, torch_dtype=torch.bfloat16)
    print("Converting model for PEFT training")
    model = get_peft_model(model, peft_config)

    n_steps = len(train_dataloader) // n_accumulation_steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=n_accumulation_steps, num_training_steps=n_steps * EPOCHS)

    model, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(model, train_dataloader, optimizer, lr_scheduler)

    is_ds_zero3 = False
    if getattr(accelerator.state, 'deepspeed_plugin', None):
        is_ds_zero3 = accelerator.state.deepspeed_plugin.zero_stage == 3
        print('using zero stage 3')

    # Training loop
    for epoch in range(EPOCHS):
        with TorchTracemalloc() as tracemalloc:
            model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
