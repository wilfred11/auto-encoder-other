import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def train(model,train_set, test_set, batch_size, training_iterations, evaluation_iterations, verbose):
    print("Training Model!")
    print(model)

    ### Set the Device ###
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ### Define the Model and Place on Device ###
    model = model.to(device)

    ### Set the Dataloaders ###
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    ### Set the Optimizer ###
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    ### Some List for logging ###
    train_loss = []
    evaluation_loss = []
    train_losses = []
    evaluation_losses = []

    ### List to store Encoded Data when Evaluating ###
    encoded_data_per_eval = []

    ### Create a Progress Bar ###
    pbar = tqdm(range(training_iterations))

    train = True
    step_counter = 0
    while train:

        for images, labels in trainloader:

            images = images.to(device)
            encoded, reconstruction = model(images)

            ### Simple MSE Loss ###
            loss = torch.mean((images - reconstruction) ** 2)
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step_counter % evaluation_iterations == 0:

                model.eval()
                encoded_evaluations = []

                for images, labels in testloader:
                    images = images.to(device)
                    encoded, reconstruction = model(images)
                    loss = torch.mean((images - reconstruction) ** 2)
                    evaluation_loss.append(loss.item())

                    ### Store the Encoded Image with their Labels ###
                    encoded, labels = encoded.cpu().flatten(1), labels.reshape(-1, 1)
                    encoded_evaluations.append(torch.cat((encoded, labels), axis=-1))

                ### Store All Testing Encoded Images ###
                encoded_data_per_eval.append(torch.concatenate(encoded_evaluations).detach())

                train_loss = np.mean(train_loss)
                evaluation_loss = np.mean(evaluation_loss)

                train_losses.append(train_loss)
                evaluation_losses.append(evaluation_loss)

                if verbose:
                    print("Training Loss", train_loss)
                    print("Evaluation Loss", evaluation_loss)

                ### Reset For Next Evaluation ###
                train_loss = []
                evaluation_loss = []

                model.train()

            step_counter += 1
            pbar.update(1)

            if step_counter >= training_iterations:
                print("Completed Training!")
                train = False
                break

    ### Store All Encoded Data as Numpy Arrays for each Eval Iteration ###
    encoded_data_per_eval = [np.array(i) for i in encoded_data_per_eval]

    print("Final Training Loss", train_losses[-1])
    print("Final Evaluation Loss", evaluation_losses[-1])

    return model, train_losses, evaluation_losses, encoded_data_per_eval