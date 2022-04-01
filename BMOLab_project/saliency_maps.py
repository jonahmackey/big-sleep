import torch
from torchvision import io, utils
import torch.nn as nn
from torch.autograd.functional import jacobian

from clip import load, tokenize
from utils import *


comp_text = "a crab walking through a field of grass"
# bg_text = "a photo of an empty field of grass"
bg_text = "a crab walking"

perceptor, normalize_image = load("ViT-B/32", jit=False)


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

    image_into = get_cutouts(img=img,
                             num_cutouts=num_cutouts,
                             legal_cutouts=legal_cutouts,
                             one_resize=False)  # shape (num_cutouts, 3, 224, 224)
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