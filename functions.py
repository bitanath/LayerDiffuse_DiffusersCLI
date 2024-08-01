import torch
import subprocess
import numpy as np

from PIL import Image

def print_memory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.free', '--format=csv'], stdout=subprocess.PIPE)
    print(result.stdout)

def clear_cache():
    with torch.no_grad():
        torch.cuda.empty_cache()

def clear_cache_print_memory():
    with torch.no_grad():
        torch.cuda.empty_cache()
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.free', '--format=csv'], stdout=subprocess.PIPE)
    print(result.stdout)

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