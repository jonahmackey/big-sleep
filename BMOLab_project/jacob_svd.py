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
