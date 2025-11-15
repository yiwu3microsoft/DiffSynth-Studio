import torch
from PIL import Image
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download

def resize_to_multiple_of_64(image):
    """Resize image so that both width and height are multiples of 64."""
    width, height = image.size
    
    # Calculate new dimensions that are multiples of 64
    new_width = ((width + 63) // 64) * 64
    new_height = ((height + 63) // 64) * 64

    return image.resize((new_width, new_height), Image.LANCZOS), new_width, new_height

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda:0",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu"),
    ],
    redirect_common_files=False
)
pipe.enable_vram_management()

# Image-to-video

input_image = Image.open("data/examples/wan/cat_glasses.jpg")
input_image, new_width, new_height = resize_to_multiple_of_64(input_image)
video = pipe(
    prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=-1, tiled=True,
    height=new_height, width=new_width,
    input_image=input_image,
    num_frames=121,
)
save_video(video, "video3-2.mp4", fps=24, quality=5)
