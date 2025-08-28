import pickle
import random
import numpy as np
import pathlib
import torch
import ffmpeg
from matplotlib import pyplot as plt, animation
from matplotlib.animation import FFMpegFileWriter

from data import get_data
from graph import build_embedding_plot, build_embedding_animation
from interpolate import interpolate_space
from model import AEncoder, SimpleAEncoder, ConvolutionalAEncoder
from pca import pca_graph
from train import train
from tsne_graph import tsne_graph

### Seed Everything ###
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

generate_anim = False
do=7
model = AEncoder()
training_iterations= 25000
evaluation_iterations = 250


if do==1:
    name=str(model.__class__.__name__)
    #pathlib.Path("graph/"+name).mkdir(parents=True, exist_ok=True)
    pathlib.Path("intermediate/" + name).mkdir(parents=True, exist_ok=True)

    train_set, test_set = get_data()

    trained_model, train_losses, evaluation_losses, encoded_data = train(model, train_set, test_set, batch_size=64, training_iterations=training_iterations, evaluation_iterations=evaluation_iterations, verbose=True)

    with open('intermediate/'+name+'/encoded_data.pkl', 'wb') as f:
        pickle.dump(encoded_data, f)

    with open('intermediate/'+name+'/train_losses.pkl', 'wb') as f:
        pickle.dump(train_losses, f)

    with open('intermediate/'+name+'/evaluation_losses.pkl', 'wb') as f:
        pickle.dump(evaluation_losses, f)

    torch.save(trained_model.state_dict(), "intermediate/"+name+"/model")

if do==2:
    name=str(model.__class__.__name__)
    pathlib.Path("graph/" + name).mkdir(parents=True, exist_ok=True)

    with open('intermediate/'+name+'/encoded_data.pkl', 'rb') as f:
        encoded_data = pickle.load(f)
    if generate_anim:
        build_embedding_animation(encoded_data, model_name=name)
    else:
        build_embedding_plot(encoded_data[-1], name+" Latents", model_name=name)

if do==3:
    # use only model with encoding outputs of 2 numbers
    name = str(model.__class__.__name__)
    pathlib.Path("graph/PCA").mkdir(parents=True, exist_ok=True)

    train_set, test_set = get_data()

    with open('intermediate/'+name+'/encoded_data.pkl', 'rb') as f:
        encoded_data = pickle.load(f)


    model.load_state_dict(torch.load( "intermediate/"+name+"/model", weights_only=True))
    model.eval()

    pca_graph(test_set, encoded_data, model, name)
    #tsne_graph(encoded_data)

if do==4:
    name = str(model.__class__.__name__)
    pathlib.Path("graph/TSNE").mkdir(parents=True, exist_ok=True)

    train_set, test_set = get_data()

    with open('intermediate/' + name + '/encoded_data.pkl', 'rb') as f:
        encoded_data = pickle.load(f)

    #model.load_state_dict(torch.load("intermediate/" + name + "/model", weights_only=True))
    #model.eval()

    tsne_graph(encoded_data, name)

if do==5:
    train_set, test_set = get_data()

    conv_model = ConvolutionalAEncoder()
    ae_model = AEncoder()

    conv_name = str(conv_model.__class__.__name__)
    ae_name = str(ae_model.__class__.__name__)

    generated_index = 0
    image, label = test_set[generated_index]

    ae_model.load_state_dict(torch.load("intermediate/" + ae_name + "/model", weights_only=True))
    conv_model.load_state_dict(torch.load("intermediate/" + conv_name + "/model", weights_only=True))
    ae_model.eval()
    conv_model.eval()
    _, ae_reconstructed = ae_model(image.unsqueeze(0).to(device))
    _, conv_reconstructed = conv_model(image.unsqueeze(0).to(device))

    ae_reconstructed = ae_reconstructed.to("cpu").detach().numpy()
    conv_reconstructed = conv_reconstructed.to("cpu").detach().numpy()

    fig, ax = plt.subplots(1, 3)

    ax[0].imshow(image.squeeze(), cmap="gray")
    ax[0].set_title("Original Image")
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    ax[0].axis("off")

    ax[1].imshow(ae_reconstructed.squeeze(), cmap="gray")
    ax[1].set_title("Autoencoder")
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[1].axis("off")

    ax[2].imshow(conv_reconstructed.squeeze(), cmap="gray")
    ax[2].set_title("Conv AutoEncoder")
    ax[2].set_xticklabels([])
    ax[2].set_yticklabels([])
    ax[2].axis("off")

    plt.savefig("graph/compare-aencoder-convencoder-original.png")

if do==7:
    ae_model = AEncoder()
    ae_name = str(ae_model.__class__.__name__)

    ae_model.load_state_dict(torch.load("intermediate/" + ae_name + "/model", weights_only=True))

    with open('intermediate/' + ae_name + '/encoded_data.pkl', 'rb') as f:
        encoded_data = pickle.load(f)

    final_embeddings = encoded_data[-1]

    avg_digit_embeddings = []

    for i in range(10):
        avg_embeddings = np.median(final_embeddings[final_embeddings[:, 2] == i][:, :2], axis=0)
        avg_digit_embeddings.append(avg_embeddings)

    avg_digit_embeddings = torch.tensor(np.array(avg_digit_embeddings))
    pred_images = ae_model.forward_dec(avg_digit_embeddings.to("cpu"))

    fig, axes = plt.subplots(1, 10, figsize=(15, 5))
    for idx, img in enumerate(pred_images):
        img = img.squeeze().detach().cpu().numpy()
        axes[idx].imshow(img, cmap="gray")
        axes[idx].set_xticklabels([])
        axes[idx].set_yticklabels([])
        axes[idx].axis("off")

    fig.subplots_adjust(wspace=0, hspace=0)

    plt.savefig("graph/generate_from_median.png")


if do==8:
    ae_model = AEncoder()
    ae_name = str(ae_model.__class__.__name__)
    ae_model.load_state_dict(torch.load("intermediate/" + ae_name + "/model", weights_only=True))
    interpolate_space(ae_model, ae_name)




