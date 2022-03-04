import cv2

from clip import load, tokenize
from big_sleep import rand_cutout

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

comp_text = "a crab walking through a field of grass"
# bg_text = "a photo of an empty field of grass"
bg_text = "a crab walking"

# perceptor, normalize_image = load("RN101", jit=False)


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
    image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
    image_embed = torch.mean(image_embed, dim=0)  # shape (512)

    return image_embed


def clip_similarity(img):
    """
    Embed img and text into CLIP and compute cosine similarity between their embeddings

    - img.shape = (1, 3, H, W)
    - entries of img lie in the range [0, 1]
    """
    image_embed = embed_image(img).unsqueeze(dim=0)  # shape (1, 512)

    # comp_tokenized_text = tokenize(comp_text).cuda()  # shape (1, 77)
    # bg_tokenized_text = tokenize(bg_text).cuda()  # shape (1, 77)
    # tokenized_text = comp_tokenized_text - bg_tokenized_text
    #
    # text_embed = perceptor.encode_text(tokenized_text)  # shape (1, 512)
    # text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

    comp_tokenized_text = tokenize(comp_text).cuda()  # shape (1, 77)
    bg_tokenized_text = tokenize(bg_text).cuda()  # shape (1, 77)

    comp_text_embed = perceptor.encode_text(comp_tokenized_text)  # shape (1, 512)
    bg_text_embed = perceptor.encode_text(bg_tokenized_text)  # shape (1, 512)

    comp_text_embed = comp_text_embed / comp_text_embed.norm(dim=-1, keepdim=True)
    bg_text_embed = bg_text_embed / bg_text_embed.norm(dim=-1, keepdim=True)

    text_embed = comp_text_embed - bg_text_embed
    text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logit_scale = perceptor.logit_scale.exp()
    clip_sim = logit_scale * image_embed @ text_embed.t()  # shape (1, 1)

    return clip_sim.squeeze()


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


def saliency_map2(title, img_fp=None, img=None, show_result=False, save_result=False):
    if img_fp is not None:
        image = io.read_image(img_fp).cuda()  # shape (3, 512, 512)
    else:
        image = img

    image = (image - image.min()) / (image.max() - image.min())  # norm to range [0, 1]
    image = torch.unsqueeze(image, dim=0).float()  # shape (1, 3, 512, 512)
    image.requires_grad = True

    # jacobian of image embedding wrt image pixels
    salience = jacobian(func=clip_similarity, inputs=image).squeeze()  # shape (3, 512, 512)
    salience = (salience * salience).sum(dim=0, keepdims=True)  # shape (1, 512, 512)
    salience_rgb = jonah_utility_func(X=salience.cpu().squeeze())  # shape (3, 512, 512)

    salience = (salience - salience.min()) / (salience.max() - salience.min())

    if show_result:
        show_image(salience, "")
        show_image(salience_rgb, "")

    if save_result:
        utils.save_image(salience.float(), f"Images/{title}/saliency_gray.png")
        utils.save_image(salience_rgb.float(), f"Images/{title}/saliency_rgb.png")

    return salience


def get_scaled_jacob(img_fp=None, img=None, jacob=None):
    if jacob is None:
        J = get_jacob(img_fp, img)  # shape (512, 3, 512, 512)
    else:
        J = jacob

    J_ = torch.sqrt((J * J).sum(dim=1))  # shape (512, 512, 512)
    J2 = nn.functional.normalize(J_, dim=0)  # shape (512, 512, 512)

    return J2


def project_J_on_T(img_fp=None, img=None, text=None):

    # Get scaled jacobian
    J_scaled = get_scaled_jacob(img_fp, img)  # shape (512, 512, 512)

    # Get image embed
    if img_fp is not None:
        image = io.read_image(img_fp).cuda()  # shape (3, 512, 512)
    else:
        image = img.cuda()

    image = (image - image.min()) / (image.max() - image.min())  # norm to range [0, 1]
    image = torch.unsqueeze(image, dim=0).float()  # shape (1, 3, 512, 512)
    image.requires_grad = True

    image_embed = embed_image(image)  # shape (1, 512)

    # get text embed
    tokenized_text = tokenize(text).cuda()  # shape (1, 77)
    text_embed = perceptor.encode_text(tokenized_text)
    text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)  # shape (1, 512)

    T = text_embed - image_embed  # shape (1, 512)
    # T = T / T.norm(dim=-1, keepdim=True)

    J_on_T = (J_scaled.permute(1, 2, 0) * T).sum(dim=-1)  # shape (512, 512)

    show_image(J_on_T.unsqueeze(dim=0), "")

    return J_on_T


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
    U = torch.from_numpy(U)

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

    return U, S, Vt


def clip_dream(img_fp, title, scaling_term=5, num_iter=10, show_result=False):
    image = io.read_image(img_fp)  # shape (3, 512, 512)
    print("input image range: ", image.min(), image.max())
    image = (image - image.min()) / (image.max() - image.min())

    if show_result:
        show_image(image, "Step 0")

    for i in range(num_iter):
        U, S, Vt = jacob_svd(title=title,
                             img=image,
                             view_sing_values=False,
                             view_top_k=0,
                             save_result=False,
                             show_result=False)

        # view singular values
        plt.figure(figsize=(10, 10))

        x = np.arange(start=1, stop=S.shape[0] + 1, step=1)
        plt.scatter(x, S)
        plt.xlabel("Rank")
        plt.ylabel("Value")
        plt.title(f"Singular Values of Jacobian iter {i+1}")

        plt.savefig(f"Images/{title}/SV_iter{i+1}.png")

        if show_result:
            plt.show()

        # get top singular vector
        PC1 = Vt[0]  # shape (3, 512, 512)
        print("PC1 max min: ", PC1.max(), PC1.min())

        # visualize top singular vector
        vmin, vmax = torch.quantile(input=PC1.abs(), q=torch.tensor([0., 0.999]))

        # Plot the RGB channels of PC separately (yellow = 1, purple = 0)
        fig = plt.figure(figsize=(30, 10))

        fig.add_subplot(1, 3, 1)
        plt.axis("off")
        plt.title(f"red iter {i + 1}")
        plt.imshow(PC1[0], vmin=-vmax, vmax=vmax, cmap='PiYG')
        plt.colorbar(shrink=0.5)

        fig.add_subplot(1, 3, 2)
        plt.axis("off")
        plt.title(f"green iter {i + 1}")
        plt.imshow(PC1[1], vmin=-vmax, vmax=vmax, cmap='PiYG')
        plt.colorbar(shrink=0.5)

        fig.add_subplot(1, 3, 3)
        plt.axis("off")
        plt.title(f"blue iter {i + 1}")
        plt.imshow(PC1[2], vmin=-vmax, vmax=vmax, cmap='PiYG')
        plt.colorbar(shrink=0.5)

        plt.savefig(f"Images/{title}/PC1_iter{i + 1}.png")

        if show_result:
            plt.show()

        # add top singular vector to image
        image = image + scaling_term * PC1  # shape (3, 512, 512)

        if show_result:
            show_image(image, f"Step {i + 1}")

        image = (image - image.min()) / (image.max() - image.min())
        utils.save_image(image.float(), f"Images/{title}/dream_iter{i + 1}.png")

    return image


def clip_dream2(img_fp, title, scaling_term=5, num_iter=10, show_result=False):

    image = io.read_image(img_fp)  # shape (3, 512, 512)
    print("input image range: ", image.min(), image.max())
    image = (image - image.min()) / (image.max() - image.min())

    if show_result:
        show_image(image, "Step 0")

    U, S, Vt = jacob_svd(title=title,
                         img=image,
                         view_sing_values=False,
                         view_top_k=0,
                         save_result=False,
                         show_result=False)

    # view singular values
    plt.figure(figsize=(10, 10))

    x = np.arange(start=1, stop=S.shape[0] + 1, step=1)
    plt.scatter(x, S)
    plt.xlabel("Rank")
    plt.ylabel("Value")
    plt.title(f"Singular Values of Jacobian")

    plt.savefig(f"Images/{title}/SV_jacob.png")

    if show_result:
        plt.show()

    # get top singular vector
    PC1 = U[:, 0]  # shape (512)
    print("PC1 max min: ", PC1.max(), PC1.min(), PC1.shape)

    image1 = image.clone()
    image2 = image.clone()

    for i in range(num_iter):
        ### dream step for image 1 ###
        J1 = get_jacob(img=image1)  # shape 512, 3, 512, 512
        J1 = J1.cpu().detach()
        J1 = J1.view(512, 3*512*512)  # shape = (512, 3*512*512) = (512 x 786432)

        X1, _, _, _ = np.linalg.lstsq(J1, PC1, rcond=None)
        print("X1 info: ", X1.shape, X1.max(), X1.min())
        X1 = torch.from_numpy(X1).view(3, 512, 512)

        # add top singular vector to image
        image1 = image1 + scaling_term * X1  # shape (3, 512, 512)

        ### dream step for image 2 ###
        J2 = get_jacob(img=image2)  # shape 512, 3, 512, 512
        J2 = J2.cpu().detach()
        J2 = J2.view(512, 3 * 512 * 512)  # shape = (512, 3*512*512) = (512 x 786432)

        X2, _, _, _ = np.linalg.lstsq(J2, -PC1, rcond=None)
        print("X2 info: ", X2.shape, X2.max(), X2.min())
        X2 = torch.from_numpy(X2).view(3, 512, 512)

        # add top singular vector to image
        image2 = image2 + scaling_term * X2  # shape (3, 512, 512)

        # display/save results
        if show_result:
            show_image(image1, f"Dream step {i + 1}")
            show_image(image2, f"neg Dream step {i + 1}")

        image1 = (image1 - image1.min()) / (image1.max() - image1.min())
        utils.save_image(image1.float(), f"Images/{title}/dream_iter{i + 1}.png")

        image2 = (image2 - image2.min()) / (image2.max() - image2.min())
        utils.save_image(image2.float(), f"Images/{title}/neg_dream_iter{i + 1}.png")

    return image


def reset_model_layers(model, model_name, layers):
    if model_name == "RN101":
        for name, layer in list(model.visual.named_children()):
            if name in layers and hasattr(layer, "reset_parameters"):
                print(f'Reset trainable parameters (in {model_name}) of layer = {name}\n')
                layer.reset_parameters()

            if name.startswith("layer"):
                for subname, sublayer in list(layer.named_children()):
                    full_subname = name + ":" + subname

                    if full_subname in layers:  # reset bottleneck layer
                        print(f'Reset trainable parameters (in {model_name}) of sublayer = {full_subname}')
                        reset_block_layer(sublayer, full_subname, model_name)

            if name == "attnpool" and name in layers:
                print(f'Reset trainable parameters (in {model_name}) of sublayer = {name}')
                reset_block_layer(layer, name, model_name)

    if model_name == "ViT":
        for name, layer in list(model.visual.named_children()):
            if name in layers and hasattr(layer, "reset_parameters"):
                print(f'Reset trainable parameters (in {model_name}) of layer = {name}\n')
                layer.reset_parameters()

            if name == "transformer":
                resblocks, resblocks_layers = list(layer.named_children())[0]

                for subname, sublayer in list(resblocks_layers.named_children()):
                    full_subname = name + ":" + subname

                    if full_subname in layers:  # reset residualattentionblock layer
                        print(f'Reset trainable parameters (in {model_name}) of sublayer = {full_subname}')
                        reset_block_layer(sublayer, full_subname, model_name)


def reset_block_layer(block, block_name, model_name):
    for name, layer in list(block.named_children()):
        if hasattr(layer, "reset_parameters"):
            print(f'Reset trainable parameters (in {model_name} block {block_name}) of sublayer = {name}')
            layer.reset_parameters()

        if name == "downsample":
            for subname, sublayer in list(layer.named_children()):
                if hasattr(sublayer, "reset_parameters"):
                    print(f'Reset trainable parameters (in {model_name} block {block_name}) of sublayer = downsample:{subname}')
                    sublayer.reset_parameters()

        if name == "attn":
            for subname, sublayer in list(layer.named_children()):
                if hasattr(sublayer, "reset_parameters"):
                    print(f'Reset trainable parameters (in {model_name} block {block_name}) of sublayer = attn:{subname}')
                    sublayer.reset_parameters()

        if name == "mlp":
            for subname, sublayer in list(layer.named_children()):
                if hasattr(sublayer, "reset_parameters"):
                    print(f'Reset trainable parameters (in {model_name} block {block_name}) of sublayer = mlp:{subname}')
                    sublayer.reset_parameters()
    print("")


if __name__ == "__main__":
    print("blah")

    perceptor, normalize_image = load("ViT-B/32", jit=False)

    dream = clip_dream2(img_fp="Images/dog.jpg", title="dream_dog_m2_vit_s=01", scaling_term=0.1, num_iter=20)

    # rn101_trainable_layers = ["conv1", "conv2", "conv3", "layer1:0", "layer1:1",
    #                           "layer1:2", "layer2:0", "layer2:1", "layer2:2", "layer2:3", "layer3:0", "layer3:1",
    #                           "layer3:2", "layer3:3", "layer3:4", "layer3:5", "layer3:6", "layer3:7", "layer3:8",
    #                           "layer3:9", "layer3:10", "layer3:11", "layer3:12", "layer3:13", "layer3:14", "layer3:15",
    #                           "layer3:16", "layer3:17", "layer3:18", "layer3:19", "layer3:20", "layer3:21", "layer3:22",
    #                           "layer4:0", "layer4:1", "layer4:2", "attnpool"]
    #
    # vit_trainable_layers = ["conv1", "ln_pre", "transformer:0", "transformer:1", "transformer:2", "transformer:3",
    #                         "transformer:4", "transformer:5", "transformer:6", "transformer:7", "transformer:8",
    #                         "transformer:9", "transformer:10", "transformer:11", "ln_post"]
    #
    # # cascading randomization
    # for i in range(len(rn101_trainable_layers) - 1):
    #     # get clip model
    #     perceptor, normalize_image = load("RN101", jit=False)
    #
    #     layers = rn101_trainable_layers[i:]
    #
    #     # reset model layer
    #     reset_model_layers(model=perceptor, model_name="RN101", layers=layers)
    #
    #     # compute saliency maps
    #     jacob, saliency_gray, saliency_rgb = saliency_map(title="", img_fp="Images/crab.png")
    #
    #     # save saliency maps
    #     utils.save_image(saliency_gray.float(),
    #                      f"Images/sanity_checks/RN101/cascading_randomization/saliency_gray_{layers[0]}_to_{layers[-1]}.png")
    #     utils.save_image(saliency_rgb.float(),
    #                      f"Images/sanity_checks/RN101/cascading_randomization/saliency_rgb_{layers[0]}_to_{layers[-1]}.png")
    #
    #     print(f"computed cascading randomized saliency for {layers[0]} to {layers[-1]}\n")

    # perceptor, normalize_image = load("RN101", jit=False)
    # jacob, saliency_gray, saliency_rgb = saliency_map(title="", img_fp="Images/crab.png", show_result=True)
