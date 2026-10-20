"""Native SS support and calibrated RGBA inputs for downstream cascade runs."""
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


def prepared_views(images, image_size, device):
    frames=[]
    for image in images:
        rgba=np.asarray(image.convert('RGBA').resize((image_size,image_size),Image.Resampling.LANCZOS)).copy()
        value=torch.from_numpy(rgba).float()/255.
        frames.append(value[...,:3].permute(2,0,1)*value[...,3].unsqueeze(0))
    return torch.stack(frames).unsqueeze(0).to(device)


def native_shape_support(occupancy):
    value=np.asarray(occupancy)
    if value.shape!=(64,64,64) or value.dtype!=np.bool_ or not value.any():
        raise ValueError('nonempty actual native SS64 binary prediction required')
    pooled=F.max_pool3d(torch.from_numpy(value.copy())[None,None].float(),2,2)>0.5
    return torch.argwhere(pooled)[:,[0,2,3,4]].int()
