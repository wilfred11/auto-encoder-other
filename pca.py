import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def pca_graph(test_set, linear_encoded_data, model, name):
    data = np.array(torch.cat([d[0].flatten(1) for d in test_set], dim=0))
    labels = np.array([d[1] for d in test_set]).reshape(-1,1)

    ### PCA Encodings ###
    pca = PCA(n_components=2)
    pca_encoding = pca.fit_transform(data)
    pca_encoding = np.concatenate((pca_encoding, labels), axis=-1)
    pca_encoding = pd.DataFrame(pca_encoding, columns=["x", "y", "class"])
    pca_encoding = pca_encoding.sort_values(by="class")
    pca_encoding["class"] = pca_encoding["class"].astype(int).astype(str)


    ### Grab Last Linear Encodings ###
    linauto_encoding = linear_encoded_data[-1]
    linauto_encoding = pd.DataFrame(linauto_encoding, columns=["x", "y", "class"])
    linauto_encoding = linauto_encoding.sort_values(by="class")
    linauto_encoding["class"] = linauto_encoding["class"].astype(int).astype(str)


    ### Lets Plot it All Now! ###
    f, (ax1,ax2) = plt.subplots(1,2, figsize=(15,5))

    ### Plot PCA Plot ###
    for grouper, group in pca_encoding.groupby("class"):
        ax1.scatter(x=group["x"], y=group["y"], label=grouper, alpha=0.8, s=5)
    ax1.legend()
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("PCA Embeddings")

    for grouper, group in linauto_encoding.groupby("class"):
        ax2.scatter(x=group["x"], y=group["y"], label=grouper, alpha=0.8, s=5)
    ax2.legend()
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Linear AutoEncoder Embeddings")

    #plt.show()
    plt.savefig('graph/'+name+'/compare-pca-autoencoder.png')

    ### Principal Component Vectors ###
    pc_1, pc_2 = torch.tensor(pca.components_[0]), torch.tensor(pca.components_[1])
    print("Angle Between PCA Principal Components", torch.rad2deg(torch.acos(torch.dot(pc_1, pc_2))).item())


    ### Linear AutoEncoder Projection Vectors ###
    model_weights = model.encoder[0].weight

    weight_vector_1, weight_vector_2 = model_weights[0].cpu(), model_weights[1].cpu()
    weight_vector_1 = weight_vector_1 / torch.norm(weight_vector_1)
    weight_vector_2 = weight_vector_2 / torch.norm(weight_vector_2)

    print("Angle Between AutoEncoder Vectors",
          torch.rad2deg(torch.acos(torch.dot(weight_vector_1, weight_vector_2))).item())

