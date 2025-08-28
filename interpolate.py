import numpy as np
import torch
from matplotlib import pyplot as plt


def interpolate_space(model, ae_name, x_range=(-3, 3), y_range=(-3, 3), num_steps=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    x_space = np.linspace(x_range[0], x_range[1], num_steps)
    y_space = np.linspace(y_range[0], y_range[1], num_steps)

    points = []
    for x in x_space:
        for y in y_space:
            points.append([x, y])

    points = torch.tensor(points, dtype=torch.float32).to(device)

    ### Pass Through Model Decoder and Reshape ###
    dec = model.forward_dec(points).detach().cpu()
    dec = dec.reshape((num_steps, num_steps, *dec.shape[1:]))

    fig, ax = plt.subplots(num_steps, num_steps, figsize=(12, 12))

    for x in range(num_steps):
        for y in range(num_steps):
            img = np.array(dec[x, y].permute(1, 2, 0))
            ax[x, y].imshow(img, cmap="gray")
            ax[x, y].set_xticklabels([])
            ax[x, y].set_yticklabels([])
            ax[x, y].axis("off")

    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig("graph/"+ae_name+"/interpolate.png")




