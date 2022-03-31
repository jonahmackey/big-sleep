import torch
import math
from torch.optim import Adam
import torch.nn as nn
from torch.fft import *

from utils import *
from saliency_maps import *
from jacob_svd import *


def embed_image2(img_freq_w, img_fft_abs, eps):
    """
    Embed img into CLIP.

    - img.shape = (1, 3, H, W)
    - entries of img lie in the range [0, 1]
    """

    img = irfft2((img_fft_abs - eps) * rfft2(img_freq_w))  # shape (1, 3, 512, 512)

    # show_image(img.squeeze(), "normal img")
    num_cutouts = 150

    # image_tensor has shape 1x3xHxW
    im_size = img.shape[2]

    legal_cutouts = torch.arange(start=1, end=16, step=1, dtype=torch.float32).cuda()
    legal_cutouts = torch.round((im_size * 7) / (7 + legal_cutouts)).int()

    image_into = get_cutouts(img=img,
                             num_cutouts=num_cutouts,
                             legal_cutouts=legal_cutouts)  # shape (num_cutouts, 3, 224, 224)
    image_into = normalize_image(image_into)

    image_embed = perceptor.encode_image(image_into)  # shape (num_cutouts, 512)
    image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
    image_embed = torch.mean(image_embed, dim=0)  # shape (512)

    return image_embed


def get_embedding_function(img_fft_abs, eps):
    def embedding_func(img_freq_w):
        return embed_image2(img_freq_w, img_fft_abs, eps)
    return embedding_func


def clip_dream(img_fp, title, eps=1e-6, theta=10, sv_index=0, num_iter=10, threshold=-0.999, lr=0.01, show_result=False):
    img = io.read_image(img_fp).cuda()  # shape (3, 512, 512)
    img = (img - img.min()) / (img.max() - img.min())  # norm to range [0, 1]
    img = torch.unsqueeze(img, dim=0).float()  # shape (1, 3, 512, 512)
    if show_result:
        show_image(img.squeeze(), "Step 0")

    utils.save_image(img.float().squeeze(), f"Images/{title}/dream_iter{0}.png")

    img_fft = rfft2(img)  # shape (1, 3, 512, 257)
    img_fft_abs = img_fft.abs()  # shape (1, 3, 512, 257)
    img_freq_w = irfft2(img_fft / (img_fft_abs + eps))  # shape (1, 3, 512, 512)
    img_freq_w.requires_grad = True

    embed_image_func = get_embedding_function(img_fft_abs=img_fft_abs, eps=eps)

    # jacobian of image embedding wrt frequency weighted pixels
    J = jacobian(func=embed_image_func, inputs=img_freq_w).squeeze()  # shape (512, 3, 512, 512)

    # singular value decomposition
    U, S, Vt = jacob_svd(title=title,
                         jacob=J,
                         view_sing_values=True,
                         view_top_k=5,
                         show_result=False,
                         save_result=True)

    # get image embedding
    E = embed_image(img)

    # get singular vector
    SV = U[:, sv_index].cuda()  # shape (512)

    for i in range(1, num_iter + 1):
        # get target embedding
        target = math.cos(i * math.radians(theta)) * E + math.sin(i * math.radians(theta)) * SV

        img = push_img_to_embedding(img=img, tgt_embed=target, threshold=threshold, lr=lr)

        # display/save results
        if show_result:
            show_image(img.squeeze(), f"Dream step {i}")

        img_normed = (img - img.min()) / (img.max() - img.min())
        utils.save_image(img_normed.float().squeeze(), f"Images/{title}/dream_iter{i}_normed.png")

        img_clipped = torch.clamp(input=img, min=0, max=1)
        utils.save_image(img_clipped.float().squeeze(), f"Images/{title}/dream_iter{i}_clipped.png")


def push_img_to_embedding(img, tgt_embed, threshold=0.1, lr=0.01):  # 0.1 thresh will ensure within 6 degrees
    img.requires_grad = True
    optimizer = torch.optim.Adam(params=[img], lr=lr)

    step = 0
    loss = 10

    while loss > threshold:
        # img = img.detatch()
        # img.requires_grad = True

        # embed image
        embed = embed_image(img).float()

        # compute loss
        # loss = - (embed @ tgt_embed.t())   # negative cosine similarity
        loss = ((embed - tgt_embed) ** 2).sum()  # negative L2 distance squared

        # update image pixels
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        step += 1

        print("\nimg_range: ", img.min(), img.max())
        print("step: ", step)
        print("loss: ", loss)

        if step == 500:
            print("break optimization")
            break

    print(f"Optimized img in {step} steps")
    return img


if __name__ == "__main__":
    print("blah")

    perceptor, normalize_image = load("ViT-B/32", jit=False)

    clip_dream(img_fp="Images/bird.jpg",
               title="clip_dream/bird6",
               eps=1e-6,
               theta=15,
               sv_index=0,
               num_iter=10,
               threshold=0.1,
               lr=0.0001,
               show_result=False)
