import cv2
import kornia as K
import numpy as np
import torch
from matplotlib import colors, cm

from big_sleep import rand_cutout


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


def get_cutouts(img, num_cutouts, legal_cutouts, one_resize=False):
    """
    Helper function to get num_cutouts random cutouts from img. Each random cutouts is resized to (224, 224) after being
    sampled.

    - img.shape = (1, 3, H, W)
    """

    if one_resize:
        resized_img = K.geometry.resize(img, (224, 224), antialias=True) # shape (1, 3, 224, 224)
        return resized_img

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

