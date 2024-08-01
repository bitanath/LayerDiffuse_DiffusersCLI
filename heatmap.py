import cv2
import torch
from torch import nn
from typing import Tuple
from torchvision import transforms as T
from torchvision.models.mobilenetv3 import mobilenet_v3_large

from PIL import Image,ImageOps
import numpy as np

def overlay(heatmap,img,colormap=cv2.COLORMAP_JET):
    if not isinstance(heatmap, np.ndarray):
        heatmap = np.asarray(heatmap)

    cam = cv2.resize(heatmap, img.size)
    cam = cv2.applyColorMap(cam, colormap)        

    if not isinstance(img, np.ndarray):
        img = np.asarray(img)

    overlay = np.uint8(0.6 * img + 0.4 * cam)
    return Image.fromarray(overlay)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.preprocess = T.Compose([T.Resize(768),T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.layer_name = "features.16.0"
        self.model = mobilenet_v3_large(True).eval()
        self.hooked = {}
        
    def forward(self,x):
        hook = self.model.features[16][0].register_forward_hook(self._forward_hook)
        tensor = self.preprocess(x).unsqueeze(0)
        output = self.model(tensor)
        feature = self.hooked['output']
        h,w = output.shape
        hook.remove()
        _, _, vT = torch.linalg.svd(feature)
        v1 = vT[:, :, 0, :][..., None, :]
        heatmap = feature @ v1.repeat(1, 1, v1.shape[3], 1)
        heatmap = heatmap.sum(1)
        heatmap -= heatmap.min()
        heatmap = heatmap / heatmap.max() * 255
        return heatmap
    
    def _forward_hook(self, module, inputs: Tuple[torch.Tensor], outputs):
        self.hooked['output'] = outputs


def heatmap(im):
    img = im.convert("RGB")
    model = Net().to('cuda')
    heatmap = model(T.ToTensor()(img).to('cuda')).cpu().detach().numpy().transpose(1, 2, 0).astype("uint8")
    im_pred = ImageOps.invert(Image.fromarray(heatmap.squeeze(),"L").resize(img.size))
    overlaid = overlay(heatmap,img)
    return overlaid