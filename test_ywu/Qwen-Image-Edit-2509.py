from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from PIL import Image
import torch
from diffusers.utils import load_image

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda:0",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)
pipe.enable_vram_management()

# image_1 = pipe(prompt="一位少女", seed=0, num_inference_steps=40, height=1328, width=1024) # 43GB, 03:08
# image_1.save("image1.jpg")

# image_2 = pipe(prompt="一位老人", seed=0, num_inference_steps=40, height=1328, width=1024) # 43GB, 03:00
# image_2.save("image2.jpg")

# prompt = "生成这两个人的合影"
# edit_image = [Image.open("image1.jpg"), Image.open("image2.jpg")]
# image_3 = pipe(prompt, edit_image=edit_image, seed=1, num_inference_steps=40, height=1328, width=1024, edit_image_auto_resize=True) # 44GB, 06:17
# image_3.save("image3.jpg")

image1 = load_image("https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/man.png")
image1 = image1.convert("RGB")
image2 = load_image("https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/puppy.png")
image2 = image2.convert("RGB")
image3 = load_image("https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/sofa.png")
image3 = image3.convert("RGB")

prompt = "Let the man in image 1 lie on the sofa in image 3, and let the puppy in image 2 lie on the floor to sleep."
inputs = {
    "edit_image": [image1, image2, image3],
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
output_image.save(f"./results/qwen-image-edit-2509.png")
