import torch
import matplotlib.pyplot as plt

from utils import *
from saliency_maps import *


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
        x = np.arange(start=1, stop=S.shape[0] + 1, step=1)

        fig = plt.figure(figsize=(20, 10))

        # singular values
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.scatter(x, S, marker=".")

        ax1.set_ylabel("Value")
        ax1.set_xlabel("Rank")
        plt.title("Singular Values of Jacobian")

        # singular values (log)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.scatter(x, S, marker=".")
        ax2.set_yscale('log')

        ax2.set_ylabel("Value (log)")
        ax2.set_xlabel("Rank")
        plt.title("Singular Values of Jacobian (log)")

        plt.savefig(f"Images/{title}/SV_jacob.png")

        if show_result:
            plt.show()

    # view top k principal components
    Vt = torch.from_numpy(Vt).view(512, 3, im_size, im_size)  # shape (512, 3, 512, 512)
    U = torch.from_numpy(U)

    if view_top_k > 0:
        for i in range(view_top_k):
            PC = Vt[i]  # torch.Size([3, 512, 512])
            vmin, vmax = torch.quantile(input=PC.abs(), q=torch.tensor([0., 1.]))

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
