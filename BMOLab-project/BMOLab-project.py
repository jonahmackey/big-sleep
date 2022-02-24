import cv2

from big_sleep.clip import load
from big_sleep.big_sleep import rand_cutout

import torch
from torch.autograd.functional import jacobian
import torch.nn as nn
import torch.nn.functional as F
from torchvision import io, utils

# from datetime import datetime
from matplotlib import colors, cm
import matplotlib.pyplot as plt
import numpy as np
# from collections import OrderedDict

# import sklearn
# from sklearn.cluster import KMeans

import kornia as K

# COMP TEXT "a crab walking through a field of grass"
# BG_TEXT "a photo of an empty field of grass"


# ---Misc Helpers---
def show_image(img, text):
    """
    Display pytorch tensor img with title title.

    - img.shape = (3, H, W) or (1, H, W)
    """
    if (img.shape[0] != 1) and (img.shape[0] != 3):
        raise Exception("Please provide PyTorch tensor with shape 3xHxW or 1xHxW")

    img = img.detach()
    img = img.permute(1, 2, 0)  # switch from CxHxW to HxWxC

    if img.shape[2] == 3:
        img = img[:, :, [2, 1, 0]]  # switch from RGB ot BGR
    else:
        img = torch.cat((img, img, img), dim=2)

    # normalize image to (0, 1)
    img = (img - img.min()) / (img.max() - img.min())
    img = 255 * img.cpu()

    np_image = np.array(img)

    cv2.imshow(text, np_image.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---Get CLIP---
perceptor, normalize_image = load("ViT-B/32", jit=False)
# "RN101", "ViT-B/32"


def get_cutouts(img, num_cutouts, legal_cutouts):
    """
    Helper function to get num_cutouts random cutouts from img. Each random cutouts is resized to (224, 224) after being
    sampled.

    - img.shape = (1, 3, H, W)
    """
    cutouts = []

    for i in range(num_cutouts):
        # get legal cutout size
        size = int(512 * torch.zeros(1, ).normal_(mean=.8, std=.3).clip(.5, .95))
        size = legal_cutouts[torch.argmin(torch.abs(legal_cutouts - size))].cpu().item()

        # get random cutout of given size
        random_cutout = rand_cutout(img, size, center_bias=False).cuda()  # shape (1, 3, size, size)

        # up/down sample to 224x224
        random_cutout = K.geometry.resize(random_cutout, (224, 224), antialias=True)  # shape (1, 3, 224, 224)

        cutouts.append(random_cutout)

    cutouts = torch.cat(cutouts)  # shape (num_cutouts, 3, 224, 224)

    return cutouts


def embed_image(img):
    """
    Embed img into CLIP.

    - img.shape = (1, 3, H, W)
    - entries of img lie in the range [0, 1]
    """
    num_cutouts = 150

    # image_tensor has shape 1x3xHxW
    im_size = img.shape[2]

    legal_cutouts = torch.arange(start=1, end=16, step=1, dtype=torch.float32).cuda()
    legal_cutouts = torch.round((im_size * 7) / (7 + legal_cutouts)).int()

    image_into = get_cutouts(img=img, num_cutouts=num_cutouts,
                             legal_cutouts=legal_cutouts)  # shape (num_cutouts, 3, 224, 224)
    image_into = normalize_image(image_into)

    image_embed = perceptor.encode_image(image_into)  # shape (num_cutouts, 512)
    image_embed = torch.mean(image_embed, dim=0)  # shape (512)

    return image_embed


# ---Get Jacobian of image embedding wrt Image pixels---
def get_jacob(img_fp=None, img=None):
    """
    Get jacobian of the image embedding with respect to image pixels, where the image
    is read from img_fp or is represented as a tensor by img.

    img.shape = (3, H, H)
    """
    if img_fp is not None:
        image = io.read_image(img_fp).cuda()  # shape (3, 512, 512)
    else:
        image = img

    image = (image - image.min()) / (image.max() - image.min())  # norm to range [0, 1]
    image = torch.unsqueeze(image, dim=0).float()  # shape (1, 3, 512, 512)
    image.requires_grad = True

    # jacobian of image embedding wrt image pixels
    J = jacobian(func=embed_image, inputs=image).squeeze()  # shape (512, 3, 512, 512)

    return J


def saliency_map(title, img_fp=None, img=None, jacob=None, show_result=False, save_result=False):
    if jacob is None:
        J = get_jacob(img_fp, img)  # shape (512, 3, 512, 512)
    else:
        J = jacob

    # get saliency maps
    J_ = torch.sqrt((J * J).sum(dim=1))  # shape (512, 512, 512)

    J1 = (1e10 * (J_ * J_).sum(dim=0))  # shape (512, 512)
    J1_ = jonah_utility_func(X=J1.cpu())  # shape (3, 512, 512)

    J1 = J1.unsqueeze(dim=0)  # shape (1, 512, 512)
    J1 = (J1 - J1.min()) / (J1.max() - J1.min())

    if show_result:
        show_image(J1, "")
        show_image(J1_, "")

    if save_result:
        utils.save_image(J1.float(), f"Images/{title}/saliency_gray.png")
        utils.save_image(J1_.float(), f"Images/{title}/saliency_rgb.png")

    return J, J1, J1_


def get_scaled_jacob(img_fp=None, img=None, jacob=None):
    if jacob is None:
        J = get_jacob(img_fp, img)  # shape (512, 3, 512, 512)
    else:
        J = jacob

    J_ = torch.sqrt((J * J).sum(dim=1))  # shape (512, 512, 512)
    J2 = nn.functional.normalize(J_, dim=0)  # shape (512, 512, 512)

    return J2


# ---utility func---
def jonah_utility_func(X, lower_quantile=0.01, upper_quantile=0.99, apply_log=True, cmap='plasma'):
    """
    - X.shape = (512, 512)
    """

    # quantile norm without clip
    input_quantiles = torch.quantile(input=X, q=torch.tensor([lower_quantile, upper_quantile]))
    vmin, vmax = input_quantiles
    J1 = (X - vmin) / (vmax - vmin)

    # add epsilon = 1 - J1.min(), apply log
    if apply_log:
        J1 = torch.log(J1 + 1. - J1.min())  # smaller epsilon

    # map to RGB
    J1_quantiles = torch.quantile(input=J1, q=torch.tensor([lower_quantile, upper_quantile]))
    vmin = float(J1_quantiles[0])
    vmax = float(J1_quantiles[1])

    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)  # same thing as line 9
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)

    J1 = scalar_map.to_rgba(x=J1)  # shape (512, 512, 4)
    J1 = torch.from_numpy(J1).permute(2, 0, 1)[:3].float()  # shape (3, 512, 512)

    return J1


# ---SVD on jacobian---
def jacob_svd(title, img_fp=None, img=None, jacob=None, view_sing_values=True, view_top_k=10, show_result=False,
              save_result=False):
    if jacob is None:
        J = get_jacob(img_fp, img)  # shape (512, 3, 512, 512)
    else:
        J = jacob

    im_size = J.shape[2]

    J = J.cpu().detach()
    J = J.view(512, 3 * im_size * im_size).numpy()  # shape = (512, 3*512*512) = (512 x 786432)

    (U, S, Vt) = np.linalg.svd(J, full_matrices=False)

    # U.shape = (512, 512)
    # S.shape = (512)
    # Vt.shape = (512, 786432)

    # plot singular values
    if view_sing_values:
        plt.figure(figsize=(10, 10))

        x = np.arange(start=1, stop=S.shape[0] + 1, step=1)
        plt.scatter(x, S)
        plt.xlabel("Rank")
        plt.ylabel("Value")
        plt.title("Singular Values of Jacobian")

        plt.savefig(f"Images/{title}/SV_jacob.png")

        if show_result:
            plt.show()

    # view top k principal components
    Vt = torch.from_numpy(Vt).view(512, 3, im_size, im_size)  # shape (512, 3, 512, 512)

    if view_top_k > 0:
        for i in range(view_top_k):
            PC = Vt[i]  # torch.Size([3, 512, 512])
            vmin, vmax = torch.quantile(input=PC.abs(), q=torch.tensor([0., 0.999]))

            # Plot the RGB channels of PC separately (yellow = 1, purple = 0)
            fig = plt.figure(figsize=(30, 10))

            fig.add_subplot(1, 3, 1)
            plt.axis("off")
            plt.title("red")
            plt.imshow(PC[0], vmin=-vmax, vmax=vmax, cmap='PiYG')
            plt.colorbar(shrink=0.5)

            fig.add_subplot(1, 3, 2)
            plt.axis("off")
            plt.title("green")
            plt.imshow(PC[1], vmin=-vmax, vmax=vmax, cmap='PiYG')
            plt.colorbar(shrink=0.5)

            fig.add_subplot(1, 3, 3)
            plt.axis("off")
            plt.title("blue")
            plt.imshow(PC[2], vmin=-vmax, vmax=vmax, cmap='PiYG')
            plt.colorbar(shrink=0.5)

            if save_result:
                plt.savefig(f"Images/{title}/PC_rgb{i + 1}.png")

            if show_result:
                plt.show()

            # channel dim normalized
            PC = torch.sqrt((PC * PC).sum(dim=0)).unsqueeze(dim=0)  # torch.Size([1, 512, 512])
            PC = (PC - PC.min()) / (PC.max() - PC.min())

            if show_result:
                show_image(PC, f"principal component {i + 1}")

            if save_result:
                utils.save_image(PC.float(), f"Images/{title}/PC_norm{i + 1}.png")

    return S, Vt


def clip_dream(img_fp, title, scaling_term=0.1, num_iter=5):
    image = io.read_image(img_fp)  # shape (3, 512, 512)
    show_image(image, "Step 0")

    for i in range(num_iter):
        iter_title = title + f"{i + 1}"
        S, Vt = jacob_svd(title=iter_title,
                          img=image,
                          view_sing_values=True,
                          view_top_k=1,
                          save_result=True,
                          show_result=True)

        # get top principal component
        PC1 = Vt[0]  # shape (3, 512, 512)

        # add top singular vector to image
        image = image + scaling_term * PC1  # shape (3, 512, 512)

        show_image(image, f"Step {i + 1}")
        utils.save_image(image.float(), f"Images/dream/dream_iter{i + 1}.png")

    return image


if __name__ == "__main__":
    print("blah")

    jacob, saliency_gray, saliency_rgb = saliency_map(title="ViT",
                                                      img_fp="Images/crab.png",
                                                      show_result=True,
                                                      save_result=True)

    print("obtained salience")

    s, vt = jacob_svd(title="ViT", jacob=jacob, view_top_k=5, show_result=True, save_result=True)
