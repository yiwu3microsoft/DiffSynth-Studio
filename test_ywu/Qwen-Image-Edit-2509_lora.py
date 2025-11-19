from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from PIL import Image
import torch
from diffusers.utils import load_image
import os
import csv

def read_tsv_to_dict(fname_csv):
    data_dict = {}
    with open(fname_csv, 'r', encoding='utf-8') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        header = next(tsv_reader)  # Read header row
        for row in tsv_reader:
            if row:  # Skip empty rows
                key = row[0]  # Use first column as key
                data_dict[key] = dict(zip(header[1:], row[1:]))  # Map remaining columns
    return data_dict

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda:0",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)
pipe.enable_vram_management()

pipe.load_lora(pipe.dit, "/tmp/output/6fb1b1d1-4842-484f-8c08-b09a3199d1f8_3c53764c/models/train/Qwen-Image-Edit-2509_bc_10k_lora/epoch-0.safetensors")

fname_csv = "data/validation_rewritten_prompt_map_real_240.csv"
data_dict = read_tsv_to_dict(fname_csv)

data_root = "data/map_real_240_simpleBG"
fname_list = os.listdir(data_root)

for idx, fname in enumerate(fname_list):
    name = fname.split(".")[0]
    if name not in data_dict:
        continue
    prompt = data_dict[name]['RewrittenPrompt']

    fname_save = f"./results/{name}.jpg"
    if os.path.exists(fname_save):
        print(f"Skip existing: {fname_save}")
        continue
    
    print(f"Processing {idx}/{len(fname_list)}: {fname} with prompt: {prompt}")

    image1 = Image.open(os.path.join(data_root, fname)).convert("RGB")

    inputs = {
        "edit_image": [image1],
        "prompt": prompt,
        "seed": 1,
        # "true_cfg_scale": 4.0,
        # "negative_prompt": " ",
        "num_inference_steps": 40,
        # "height": 1328,
        # "width": 1024,
        "edit_image_auto_resize": True
    }

    output_image = pipe(**inputs)
    output_image.save(fname_save)
