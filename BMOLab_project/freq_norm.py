import torch
from torch.fft import *

from utils import *
from saliency_maps import *
from jacob_svd import *


def norm_img_freq(image):
    eps = 1e-4

    image = (image - image.min()) / (image.max() - image.min())

    image_fft = rfft2(image)  # shape (3, 512, 257)
    image_fft2 = image_fft / (image_fft.abs() + eps)

    image_freq_weighted = irfft2(image_fft2)  # shape (3, 512, 512)
    image_freq_weighted = (image_freq_weighted - image_freq_weighted.min()) / (
                image_freq_weighted.max() - image_freq_weighted.min())

    # show_image(image_freq_weighted, text="")
    return image_freq_weighted


def jacob_freq_normed(img_fp=None, img=None, eps=1e-444):
    if img_fp is not None:
        image = io.read_image(img_fp).cuda()  # shape (3, 512, 512)
    else:
        image = img

    image = (image - image.min()) / (image.max() - image.min())  # norm to range [0, 1]

    image = torch.unsqueeze(image, dim=0).float()  # shape (1, 3, 512, 512)
    image.requires_grad = True

    # Get dE / dI
    J = get_jacob(img=image)  # shape (512, 3, 512, 512)

    # Get dI^* / dI
    J_freq = jacobian(func=norm_img_freq, inputs=image).squeeze()  # shape (3, 512, 512, 3, 512, 512)

    return J


# get image
# get I~ and store |I~| locally
# get I~*
# get I*
# computer encoding from I* to E


def embed_image2(img_freq_w):
    """
    Embed img into CLIP.

    - img.shape = (1, 3, H, W)
    - entries of img lie in the range [0, 1]
    """
    
    img = irfft2(img_fft_norm * rfft2(img_freq_w))
     
    
    num_cutouts = 150

    # image_tensor has shape 1x3xHxW
    im_size = img.shape[2]

    legal_cutouts = torch.arange(start=1, end=16, step=1, dtype=torch.float32).cuda()
    legal_cutouts = torch.round((im_size * 7) / (7 + legal_cutouts)).int()

    image_into = get_cutouts(img=img, num_cutouts=num_cutouts,
                             legal_cutouts=legal_cutouts)  # shape (num_cutouts, 3, 224, 224)
    image_into = normalize_image(image_into)

    image_embed = perceptor.encode_image(image_into)  # shape (num_cutouts, 512)
    image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
    image_embed = torch.mean(image_embed, dim=0)  # shape (512)

    return image_embed


if __name__ == "__main__":
    print("blah")

    perceptor, normalize_image = load("ViT-B/32", jit=False)

    image = io.read_image("Images/crab.png").cuda()  # shape (3, 512, 512)
    image = (image - image.min()) / (image.max() - image.min())  # norm to range [0, 1]
    image = torch.unsqueeze(image, dim=0).float()  # shape (1, 3, 512, 512)

    eps = 1e-4

    # Get I~
    image_fft = rfft2(image)  # shape (1, 3, 512, 257) I~

    img_fft_norm = image_fft.abs().float() # shape (1, 3, 512, 257) |I~|

    image_freq_weighted = irfft2(image_fft / (image_fft.abs() + eps)).squeeze().float()  # shape (3, 512, 512) |I*|
    image_freq_weighted.requires_grad = True
    image_freq_weighted = (image_freq_weighted - image_freq_weighted.min()) / (image_freq_weighted.max() - image_freq_weighted.min())  # norm to range [0, 1]

    # jacobian of image embedding wrt image pixels
    J = jacobian(func=embed_image2, inputs=image_freq_weighted).squeeze()  # shape (512, 3, 512, 512)
    
    jacob, sal_gray, sal_rgb = saliency_map(title="freq_normed", jacob=J, show_result=True, save_result=True)

    U, S, Vt = jacob_svd("freq_normed", jacob=J, view_sing_values=True, view_top_k=10, show_result=True, save_result=True)
    