from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from PIL import Image
import torch
from diffusers.utils import load_image
import os

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

pipe.load_lora(pipe.dit, "/tmp/output/6fb1b1d1-4842-484f-8c08-b09a3199d1f8_3c53764c/models/train/Qwen-Image-Edit-2509_bc_10k_lora/epoch-0.safetensors")

with open("test_ywu/edit_img_prompt.tsv", "r") as f:
    lines = f.readlines()

for img_prompt in lines:
    img_prompt = img_prompt.strip()
    if not img_prompt:
        continue
    img_name, prompt = img_prompt.split("\t")
    image1 = Image.open(os.path.join('data', img_name)).convert("RGB")

    inputs = {
        "edit_image": [image1],
        "prompt": prompt,
        "seed": 1,
        # "true_cfg_scale": 4.0,
        # "negative_prompt": " ",
        "num_inference_steps": 40,
        # "height": 1328,
        # "width": 1024,
        "edit_image_auto_resize": False
    }

    output_image = pipe(**inputs)
    output_image.save(f"./results/{img_name}")
