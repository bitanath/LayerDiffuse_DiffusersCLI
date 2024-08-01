import sys
import numpy as np
import torch
import memory_management
import safetensors.torch as sf

from PIL import Image
from diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from lib_layerdiffuse.vae import TransparentVAEDecoder, TransparentVAEEncoder
from lib_layerdiffuse.utils import download_model

sdxl_name = 'SG161222/RealVisXL_V4.0'
tokenizer = CLIPTokenizer.from_pretrained(
    sdxl_name, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(
    sdxl_name, subfolder="tokenizer_2")
text_encoder = CLIPTextModel.from_pretrained(
    sdxl_name, subfolder="text_encoder", torch_dtype=torch.float16, variant="fp16")
text_encoder_2 = CLIPTextModel.from_pretrained(
    sdxl_name, subfolder="text_encoder_2", torch_dtype=torch.float16, variant="fp16")
vae = AutoencoderKL.from_pretrained(
    sdxl_name, subfolder="vae", torch_dtype=torch.bfloat16, variant="fp16")  # bfloat16 vae
unet = UNet2DConditionModel.from_pretrained(
    sdxl_name, subfolder="unet", torch_dtype=torch.float16, variant="fp16")

default_negative = 'face asymmetry, eyes asymmetry, deformed eyes, open mouth, nsfw'

@torch.inference_mode()
def pytorch2numpy(imgs):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)
        y = y * 127.5 + 127.5
        y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.5 - 1.0
    h = h.movedim(-1, 1)
    return h


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)

# SDP
print("Setting attention processor")

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Download Mode
print("Checking downloads and downloading if necessary")
path_ld_diffusers_sdxl_attn = download_model(
    url='https://huggingface.co/lllyasviel/LayerDiffuse_Diffusers/resolve/main/ld_diffusers_sdxl_attn.safetensors',
    local_path='./models/ld_diffusers_sdxl_attn.safetensors'
)

path_ld_diffusers_sdxl_vae_transparent_encoder = download_model(
    url='https://huggingface.co/lllyasviel/LayerDiffuse_Diffusers/resolve/main/ld_diffusers_sdxl_vae_transparent_encoder.safetensors',
    local_path='./models/ld_diffusers_sdxl_vae_transparent_encoder.safetensors'
)

path_ld_diffusers_sdxl_vae_transparent_decoder = download_model(
    url='https://huggingface.co/lllyasviel/LayerDiffuse_Diffusers/resolve/main/ld_diffusers_sdxl_vae_transparent_decoder.safetensors',
    local_path='./models/ld_diffusers_sdxl_vae_transparent_decoder.safetensors'
)

# Modify
print("Loading files for SD")
sd_offset = sf.load_file(path_ld_diffusers_sdxl_attn)
sd_origin = unet.state_dict()
keys = sd_origin.keys()
sd_merged = {}

for k in sd_origin.keys():
    if k in sd_offset:
        sd_merged[k] = sd_origin[k] + sd_offset[k]
    else:
        sd_merged[k] = sd_origin[k]

print("Setting UNet in Cuda")
unet.cuda()

unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged, keys, k


transparent_encoder = TransparentVAEEncoder(path_ld_diffusers_sdxl_vae_transparent_encoder).cuda()
transparent_decoder = TransparentVAEDecoder(path_ld_diffusers_sdxl_vae_transparent_decoder).cuda()

# Pipelines

pipeline = KDiffusionStableDiffusionXLPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    unet=unet,
    scheduler=None,  # We completely give up diffusers sampling system and use A1111's method
)

print("Created Pipeline")

prompt = sys.argv[1]
negative=default_negative
num_inference_steps=25
guidance_scale=7.0
# def infer(prompt,negative=default_negative,num_inference_steps=25,guidance_scale=7.0):
with torch.inference_mode():
    torch.cuda.empty_cache()
    positive_cond, positive_pooler = pipeline.encode_cropped_prompt_77tokens(
        prompt
    )

    rng = torch.Generator(device='cuda').manual_seed(12345)

    negative_cond, negative_pooler = pipeline.encode_cropped_prompt_77tokens(negative)

    initial_latent = torch.zeros(size=(1, 4, 144, 112), dtype=unet.dtype, device=unet.device)
    latents = pipeline(
        initial_latent=initial_latent,
        strength=1.0,
        num_inference_steps=num_inference_steps,
        batch_size=1,
        prompt_embeds=positive_cond,
        negative_prompt_embeds=negative_cond,
        pooled_prompt_embeds=positive_pooler,
        negative_pooled_prompt_embeds=negative_pooler,
        generator=rng,
        guidance_scale=guidance_scale,
    ).images
    
    latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
    result_list, vis_list = transparent_decoder(vae, latents)

    for i, image in enumerate(result_list):
        Image.fromarray(image).save(f'./imgs/outputs/t2i_{i}_transparent.png', format='PNG')

    for i, image in enumerate(vis_list):
        Image.fromarray(image).save(f'./imgs/outputs/t2i_{i}_visualization.png', format='PNG')
