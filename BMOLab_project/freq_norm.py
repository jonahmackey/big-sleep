import os
import math
import torch
from torch.fft import *

from utils import *
from saliency_maps import *
from jacob_svd import *

from typing import Optional

torch.autograd.set_detect_anomaly(True)


def clip_dream(img_fp: str,
               title: str,
               num_dream_steps: int = 10,
               theta: int = 1,
               sv_index: int = 0,
               threshold: float = 0.1,
               eps: float = 1e-6,
               num_cutouts: int = 32,
               one_resize: bool = False,
               freq_reg: Optional[str] = 'norm',
               lr: float = 10,
               max_iters: int = 10,
               root: bool = True,
               rand_direction: bool = False,
               show_result: bool = False,
               make_vid: bool = False,
               show_saliency: bool = False,
               print_losses: bool = False,
               print_gradients: bool = False):
    img = io.read_image(img_fp).cuda()  # shape (3, 512, 512)
    img = img / 255  # put pixels in range [0, 1]
    img = torch.unsqueeze(img, dim=0).float()  # shape (1, 3, 512, 512)
    if show_result:
        show_image(img.squeeze(), "Step 0")

    utils.save_image(img.float().squeeze(), f"Images/{title}/dream_iter{0:>04d}.png")

    if freq_reg == 'norm':
        img_fft = rfft2(img)  # shape (1, 3, 512, 257)
        img_fft_abs = img_fft.abs()  # shape (1, 3, 512, 257)
        input_img = irfft2(img_fft / (img_fft_abs + eps))  # shape (1, 3, 512, 512)
    else:
        input_img = img.clone()
        img_fft_abs = 1

    input_img.requires_grad = True

    embed_image_func = get_embedding_function(img_fft_abs=img_fft_abs, eps=eps, num_cutouts=num_cutouts,
                                              freq_reg=freq_reg, one_resize=one_resize)

    # jacobian of image embedding wrt frequency weighted pixels
    J = jacobian(func=embed_image_func, inputs=input_img).squeeze()  # shape (512, 3, 512, 512)
    print("Got freq normed jacobian")

    if show_saliency:
        _ = saliency_map(title=title, jacob=J, show_result=True, save_result=True)

    # get image embedding
    standard_embed_func = get_embedding_function(freq_reg=False, num_cutouts=num_cutouts, one_resize=one_resize)
    E = standard_embed_func(img)  # shape (512)

    # get singular vector
    if rand_direction:
        # random unit vector
        V = torch.rand(512).half().cuda()
        V = V / (V.norm(dim=-1, keepdim=True))

        # get orthonormal vector to V
        V = V - (E @ V) * E
        V = V / (V.norm(dim=-1, keepdim=True))

    else:
        # singular value decomposition
        U, S, Vt = jacob_svd(title=title,
                             jacob=J,
                             view_sing_values=True,
                             view_top_k=sv_index + 1,
                             show_result=False,
                             save_result=True)

        V = U[:, sv_index].cuda()  # shape (512)

    # keep track off clip dream losses
    dream_losses = []
    dream_cp_iters = []  # checkpointing iterations between points
    dream_intermediate_iters = []

    # keep track of pixel range
    pixel_max = [img.max().item()]
    pixel_min = [img.min().item()]

    print("Starting optimization")
    for i in range(1, num_dream_steps+1):
        # get target
        target = math.cos(i * math.radians(theta)) * E + math.sin(i * math.radians(theta)) * V

        img, iter_losses = push_img_to_embedding(img=img,
                                                 tgt_embed=target,
                                                 threshold=threshold,
                                                 lr=lr,
                                                 max_iters=max_iters,
                                                 num_cutouts=num_cutouts,
                                                 root=root,
                                                 print_losses=print_losses,
                                                 print_gradients=print_gradients,
                                                 one_resize=one_resize)

        print(f"Reached point {i}")

        # save loss data
        dream_losses += iter_losses
        dream_cp_iters.append(len(dream_losses))
        dream_intermediate_iters.append(len(iter_losses))

        # save pixel range data
        pixel_max.append(img.max().item())
        pixel_min.append(img.min().item())

        # display/save results
        if show_result:
            show_image(img.squeeze(), f"Dream step {i}")

        # save clipped image
        img_clipped = torch.clamp(input=img, min=0, max=1)
        utils.save_image(img_clipped.float().squeeze(), f"Images/{title}/dream_iter{i:>04d}.png")

    step_avg = sum(dream_intermediate_iters) / len(dream_cp_iters)
    print("Average iter steps: ", step_avg)

    # plot losses
    x_iters = list(range(1, len(dream_losses)+1))
    plt.figure(figsize=(16, 8))
    plt.title(f"CLIP Dream optimization loss (lr={lr}, thresh={threshold}, theta={theta}, sv_index={sv_index})")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.xlim([0, len(dream_losses)+1])
    plt.ylim([0, 1.2*max(dream_losses[10:])])

    plt.plot(x_iters, dream_losses, 'bo-', lw=1.5)

    for i in range(len(dream_cp_iters)):
        plt.plot([dream_cp_iters[i], dream_cp_iters[i]], [0, 5], 'r-', lw=2, dashes=[2, 2])
        plt.plot([0, len(dream_losses)], [threshold, threshold], 'r-', lw=2, dashes=[2, 2])

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

    # plot points vs steps
    x_data2 = list(range(1, num_dream_steps+1))

    plt.figure(figsize=(8, 8))
    plt.xlabel("Points along path")
    plt.ylabel("Num steps")
    plt.title("Num iter steps vs. points")
    plt.plot(x_data2, dream_intermediate_iters, 'bo-', lw=1.5)
    plt.plot([1, num_dream_steps], [step_avg, step_avg], 'r-', lw=2, dashes=[2, 2])

    plt.savefig(f"Images/{title}/points_vs_steps.png")

    if show_result:
        plt.show()

    # create video
    if make_vid:
        make_video(frames_path=title)


def push_img_to_embedding(img, tgt_embed, threshold=0.1, lr=1., max_iters=500, num_cutouts=32, root=True,
                          print_losses=False, print_gradients=False, one_resize=False):

    img.requires_grad = True
    optimizer = torch.optim.SGD(params=[img], lr=lr)
    # optimizer = torch.optim.Adam(params=[img], lr=lr)

    step = 0
    loss = 10
    losses = []

    embed_image_func = get_embedding_function(freq_reg=False, num_cutouts=num_cutouts, one_resize=one_resize)

    while loss > threshold:

        # embed image
        embed = embed_image_func(img).float()

        # compute loss
        if root:
            loss = torch.norm(input=embed - tgt_embed, p='fro')
        else:
            loss = torch.norm(input=embed - tgt_embed, p='fro') ** 2

        losses.append(loss.item())

        # update image pixels
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # keep track of iter steps
        step += 1

        # print info
        if print_gradients:
            if step % 50 == 1:
                print(f"Gradients (step={step}): ", img.grad.norm(), img.grad.min(), img.grad.max())

        # print info
        if print_losses:
            if step % 100 == 1:
                print(f"Loss (step={step}): ", loss.item())

        # break loop after reaching max_iters iters
        if step == max_iters:
            print("break optimization")
            break

    print(f"Optimized img in {step} steps, with final loss={loss}")
    return img, losses


def make_video(frames_path, fps=5):
    os.system(f"ffmpeg -framerate {fps} -i ./Images/{frames_path}/dream_iter%04d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ./Images/{frames_path}/clip_dream.mp4")


if __name__ == "__main__":
    print("blah")

    clip_dream(img_fp="Images/bird.jpg",
               title="clip_dream2/exp6",
               num_dream_steps=25,
               theta=1,
               sv_index=2,
               threshold=0.1,  # previously was 0.002
               eps=1e-6,
               num_cutouts=32,
               one_resize=False,
               freq_reg='norm',
               lr=250,
               max_iters=1000,
               root=True,
               rand_direction=False,
               show_result=False,
               make_vid=True,
               show_saliency=True,
               print_losses=False,
               print_gradients=False)

