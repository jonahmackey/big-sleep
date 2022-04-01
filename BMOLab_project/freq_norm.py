import os
import math
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
                             legal_cutouts=legal_cutouts,
                             one_resize=False)  # shape (num_cutouts, 3, 224, 224)
    image_into = normalize_image(image_into)

    image_embed = perceptor.encode_image(image_into)  # shape (num_cutouts, 512)
    image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
    image_embed = torch.mean(image_embed, dim=0)  # shape (512)

    return image_embed


def get_embedding_function(img_fft_abs, eps):
    def embedding_func(img_freq_w):
        return embed_image2(img_freq_w, img_fft_abs, eps)
    return embedding_func


def clip_dream(img_fp,
               title,
               eps=1e-6,
               theta=10,
               sv_index=0,
               num_dream_steps=10,
               threshold=0.1,
               lr=0.01,
               show_result=False,
               make_vid=False,
               max_iters=10,
               show_saliency=False):
    img = io.read_image(img_fp).cuda()  # shape (3, 512, 512)
    img = img / 255  # put pixels in range [0, 1]
    img = torch.unsqueeze(img, dim=0).float()  # shape (1, 3, 512, 512)
    if show_result:
        show_image(img.squeeze(), "Step 0")

    utils.save_image(img.float().squeeze(), f"Images/{title}/dream_iter{0:>04d}.png")

    img_fft = rfft2(img)  # shape (1, 3, 512, 257)
    img_fft_abs = img_fft.abs()  # shape (1, 3, 512, 257)
    img_freq_w = irfft2(img_fft / (img_fft_abs + eps))  # shape (1, 3, 512, 512)
    img_freq_w.requires_grad = True

    embed_image_func = get_embedding_function(img_fft_abs=img_fft_abs, eps=eps)

    # jacobian of image embedding wrt frequency weighted pixels
    J = jacobian(func=embed_image_func, inputs=img_freq_w).squeeze()  # shape (512, 3, 512, 512)

    if show_saliency:
        _ = saliency_map(title=title, jacob=J, show_result=True)

    # singular value decomposition
    U, S, Vt = jacob_svd(title=title,
                         jacob=J,
                         view_sing_values=True,
                         view_top_k=sv_index+1,
                         show_result=False,
                         save_result=True)

    # get image embedding
    E = embed_image(img)

    # get singular vector
    SV = U[:, sv_index].cuda()  # shape (512)

    # keep track off clip dream losses
    dream_losses = []
    dream_cp_iters = []  # checkpointing iterations between points

    # keep track of pixel range
    pixel_max = [img.max().item()]
    pixel_min = [img.min().item()]

    for i in range(1, num_dream_steps+1):
        # get target embedding
        target = math.cos(i * math.radians(theta)) * E + math.sin(i * math.radians(theta)) * SV

        img, iter_losses = push_img_to_embedding(img=img, tgt_embed=target, threshold=threshold,
                                                 lr=lr, max_iters=max_iters)

        # save loss data
        dream_losses += iter_losses
        dream_cp_iters.append(len(dream_losses))

        # save pixel range data
        pixel_max.append(img.max().item())
        pixel_min.append(img.min().item())

        # display/save results
        if show_result:
            show_image(img.squeeze(), f"Dream step {i}")

        # save clipped image
        img_clipped = torch.clamp(input=img, min=0, max=1)
        utils.save_image(img_clipped.float().squeeze(), f"Images/{title}/dream_iter{i:>04d}.png")

    # plot losses
    x_iters = list(range(1, len(dream_losses)+1))
    plt.figure(figsize=(16, 8))
    plt.title(f"CLIP Dream optimization loss (lr={lr}, thresh={threshold}, theta={theta}, sv_index={sv_index})")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.xlim([0, len(dream_losses)+1])
    plt.ylim([0, 1.5*max(dream_losses)])

    plt.plot(x_iters, dream_losses, 'bo-', lw=1.5)

    for i in range(len(dream_cp_iters)):
        plt.plot([dream_cp_iters[i], dream_cp_iters[i]], [0, 5], 'r-', lw=2, dashes=[2, 2])

    plt.savefig(f"Images/{title}/clip_dream_loss.png")

    if show_result:
        plt.show()

    # plot pixel range
    x_data = list(range(len(pixel_max)))

    plt.figure(figsize=(8, 8))
    plt.xlabel("Optim points")
    plt.ylabel("Pixel value")
    plt.title("Max/min pixel value over optimization")
    plt.ylim([-5, 5])
    plt.plot(x_data, pixel_max, 'bo-', lw=1.5, label='Max pixel')
    plt.plot(x_data, pixel_min, 'ro-', lw=1.5, label='Min pixel')
    plt.legend()

    plt.savefig(f"Images/{title}/pixel_range.png")

    if show_result:
        plt.show()

    # create video
    if make_vid:
        make_video(frames_path=title)


def push_img_to_embedding(img, tgt_embed, threshold=0.1, lr=1, max_iters=50):  # 0.1 thresh will ensure within 6 degrees

    img.requires_grad = True
    optimizer = torch.optim.SGD(params=[img], lr=lr)

    step = 0
    loss = 10
    losses = []

    while loss > threshold:

        # embed image
        embed = embed_image(img).float()

        # compute loss
        loss = ((embed - tgt_embed) ** 2).sum()  # L2 distance squared
        loss = torch.norm(input=embed-tgt_embed, p='fro')
        losses.append(loss.item())

        # update image pixels
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # # print gradients
        # print("Gradients: ", img.grad.norm(), img.grad.min(), img.grad.max())

        # keep track of iter steps
        step += 1

        # break loop after reaching 500 iter steps
        if step == max_iters:
            print("break optimization")
            break

    print(f"Optimized img in {step} steps")
    return img, losses


def make_video(frames_path, fps=10):
    os.system(f"ffmpeg -framerate {fps} -i ./Images/{frames_path}/dream_iter%04d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ./Images/{frames_path}/clip_dream.mp4")


if __name__ == "__main__":
    print("blah")

    perceptor, normalize_image = load("ViT-B/32", jit=False)

    clip_dream(img_fp="Images/bird.jpg",
               title="clip_dream/bird11",
               eps=1e-6,
               theta=1,
               sv_index=0,
               num_dream_steps=8,
               threshold=0.002,
               lr=200,
               show_result=False,
               make_vid=True,
               max_iters=1000,
               show_saliency=False)

