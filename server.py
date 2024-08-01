import sys
import numpy as np
import torch
import safetensors.torch as sf
import subprocess

from io import BytesIO
import base64

from functions import print_memory,clear_cache_print_memory,clear_cache,pytorch2numpy,numpy2pytorch,resize_without_crop
from chat import interrogate_type,interrogate_good,interrogate_bad,interrogate_spelling_grammar,interrogate_colors,interrogate_complementary,interrogate_add_item,interrogate_canva_add

from PIL import Image
from diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from lib_layerdiffuse.vae import TransparentVAEDecoder, TransparentVAEEncoder
from lib_layerdiffuse.utils import download_model
from transformers import AutoModel, AutoTokenizer
from flask import Flask,request,make_response

app = Flask(__name__)
model_path = 'openbmb/MiniCPM-Llama3-V-2_5'
chatgpt_model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(dtype=torch.float16)
chatgpt_model = chatgpt_model.to(device='cuda')
chatgpt_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
chatgpt_model.eval()

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

default_negative = 'face asymmetry, eyes asymmetry, deformed eyes, open mouth, nsfw, robot eyes, distorted, bad anatomy, medium quality, blurry, blurred, low resolution'

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Download Mode

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

## Warmup the server
print("Warming up:")
clear_cache_print_memory()

sd_offset = sf.load_file(path_ld_diffusers_sdxl_attn)
sd_origin = unet.state_dict()
keys = sd_origin.keys()
sd_merged = {}

for k in sd_origin.keys():
    if k in sd_offset:
        sd_merged[k] = sd_origin[k] + sd_offset[k]
    else:
        sd_merged[k] = sd_origin[k]

unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged, keys, k

transparent_encoder = TransparentVAEEncoder(path_ld_diffusers_sdxl_vae_transparent_encoder)
transparent_decoder = TransparentVAEDecoder(path_ld_diffusers_sdxl_vae_transparent_decoder)

pipeline = KDiffusionStableDiffusionXLPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    unet=unet,
    scheduler=None,  # We completely give up diffusers sampling system and use A1111's method
)
default_keywords=", masterpiece, best quality, absurdres"

prompt = "Snake" + default_keywords
print("Now inferring with prompt ",prompt)

clear_cache_print_memory()

with torch.inference_mode():
    guidance_scale = 7.0

    rng = torch.Generator(device='cuda').manual_seed(12345)

    text_encoder.to('cuda')
    text_encoder_2.to('cuda')

    positive_cond, positive_pooler = pipeline.encode_cropped_prompt_77tokens(
        prompt
    )

    negative_cond, negative_pooler = pipeline.encode_cropped_prompt_77tokens(default_negative)

    unet.to('cuda')
    initial_latent = torch.zeros(size=(1, 4, 144, 112), dtype=unet.dtype, device=unet.device)
    latents = pipeline(
        initial_latent=initial_latent,
        strength=1.0,
        num_inference_steps=25,
        batch_size=1,
        prompt_embeds=positive_cond,
        negative_prompt_embeds=negative_cond,
        pooled_prompt_embeds=positive_pooler,
        negative_pooled_prompt_embeds=negative_pooler,
        generator=rng,
        guidance_scale=guidance_scale,
    ).images

    vae.to('cuda')
    transparent_decoder.to('cuda')
    transparent_encoder.to('cuda')
    latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
    result_list, vis_list = transparent_decoder(vae, latents)

    for i, image in enumerate(result_list):
        print("Warmed up generator: ",Image.fromarray(image))

    clear_cache_print_memory()

    msgs = [{'role': 'user', 'content': "What does this image contain?"}]
    default_params = {"stream": False, "sampling": False, "num_beams":3, "repetition_penalty": 1.2, "max_new_tokens": 1024}
    res = chatgpt_model.chat(
        image=Image.fromarray(image),
        msgs=msgs,
        tokenizer=chatgpt_tokenizer,
        **default_params
    )

    clear_cache_print_memory()
    print("Warmed up chatter: ",res)


@app.route('/ping', methods=['GET'])
def pinging():
    return 'pong'

@app.route('/', methods=['GET'])
def index():
    b64 = request.args.get("base64")
    if(b64):
        
        b64 = b64.replace("data:image/png;base64,","")
        print("Got image",b64)
        im = Image.open(BytesIO(base64.b64decode(b64)))
        msgs = [{'role': 'user', 'content': "How would you describe this image? Be as succinct as possible."}]
        default_params = {"stream": False, "sampling": False, "num_beams":3, "repetition_penalty": 1.2, "max_new_tokens": 128}
        res = chatgpt_model.chat(
            image=im,
            msgs=msgs,
            tokenizer=chatgpt_tokenizer,
            **default_params
        )

        clear_cache_print_memory()
        return res
    else:
        return "No image"
    
@app.route('/check', methods=['GET'])
def check():
    prompt = request.args.get("prompt")
    if(prompt):
        prompt = prompt + default_keywords
        positive_cond, positive_pooler = pipeline.encode_cropped_prompt_77tokens(prompt)

        negative_cond, negative_pooler = pipeline.encode_cropped_prompt_77tokens(default_negative)

        initial_latent = torch.zeros(size=(1, 4, 144, 112), dtype=unet.dtype, device=unet.device)
        latents = pipeline(
            initial_latent=initial_latent,
            strength=1.0,
            num_inference_steps=25,
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
            img = Image.fromarray(image)
        clear_cache_print_memory()
        return "Generated image"+str(img.size())
    else:
        return "No prompt"

@app.route('/metadata', methods=['POST'])
def json_example():
    request_data = request.get_json() #TODO get the base64 encoded thumbnail image of the design here
    request_headers = request.headers #TODO check the headers for requisite identification

    #return a dict to return a JSON by default
    response = make_response({
        "test": "this"
    })
    response.headers.set("Content-Type","application/json")
    return response



app.run(debug=True, host='localhost', port=8080, use_reloader=False)