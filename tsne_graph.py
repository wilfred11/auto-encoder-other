import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


def tsne_graph(conv_encoded_data_per_eval, name):
    conv_ae_encoding = conv_encoded_data_per_eval[-1]

    conv_ae_features = conv_ae_encoding[:, :-1]
    labels = conv_ae_encoding[:, -1].reshape(-1, 1)
    tsne = TSNE(2, n_jobs=-1)
    conv_ae_compressed = tsne.fit_transform(X=conv_ae_features)

    conv_ae_encoding = np.hstack((conv_ae_compressed, labels))
    conv_ae_encoding = pd.DataFrame(conv_ae_encoding, columns=["x", "y", "class"])
    conv_ae_encoding = conv_ae_encoding.sort_values(by="class")
    conv_ae_encoding["class"] = conv_ae_encoding["class"].astype(int).astype(str)

    leg = []
    for grouper, group in conv_ae_encoding.groupby("class"):
        plt.scatter(x=group["x"], y=group["y"], label=grouper, alpha=0.8, s=5)
        leg.append(grouper)

    plt.legend(leg, loc="lower right")
    plt.savefig("graph/TSNE/" + name + ".png")
