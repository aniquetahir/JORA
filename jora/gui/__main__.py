import gradio as gr
import tkinter as tk
from tkinter import filedialog
import time

def select_directory():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()
    return folder_path

def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path

def dummy_train():
    print('Training')
    time.sleep(10)
    return 'done'

with gr.Blocks(analytics_enabled=False) as ui_component:
    with gr.Row():
        txt_model_size = gr.components.Dropdown(["7B", "13B", "70B"], label="Model Size")
    with gr.Row():
        txt_hf = gr.components.Textbox("hf_models/path", label="Llama-2 pretrained path", lines=1)
        btn_hf = gr.components.Button('Set Llama-2 path')
        btn_hf.click(select_directory, outputs=txt_hf)
    with gr.Row():
        txt_jax = gr.components.Textbox("jax_model.pkl", label="JAX model path", lines=1)
        btn_jax = gr.components.Button('Set JAX model path')
        btn_jax.click(select_file, outputs=txt_jax)
    with gr.Row():
        txt_dataset = gr.components.Textbox("dataset.json", label="Dataset path (Alpaca format)", lines=1)
        btn_dataset = gr.components.Button('Set dataset path')
        btn_dataset.click(select_file, outputs=txt_dataset)
    with gr.Row():
        txt_checkpoint = gr.components.Textbox("checkpoints", label="Checkpoints path", lines=1)
        btn_checkpoint = gr.components.Button('Set checkpoints path')
        btn_checkpoint.click(select_directory, outputs=txt_checkpoint)
    with gr.Row():
        # txt_done = gr.components.Text("", lines=1)
        lbl_train = gr.components.Label("Start Training -->", show_label=False)
        # txt_done = gr.components.Textbox("", lines=1, show_label=False)
        btn_train = gr.components.Button('Train')
        btn_train.click(dummy_train, outputs=lbl_train)


if __name__ == "__main__":
    ui_component.launch()