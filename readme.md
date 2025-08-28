## Autoencoders and more

An autoencoder is a type of artificial neural network used to learn efficient codings of unlabeled data (unsupervised learning). An autoencoder learns two functions: an encoding function that transforms the input data, and a decoding function that recreates the input data from the encoded representation. The autoencoder learns an efficient representation (encoding) for a set of data, typically for dimensionality reduction, to generate lower-dimensional embeddings for subsequent use by other machine learning algorithms.

### A very simple autoencoder

In an autoencoder two parts need to be present, the encoder.
```
self.encoder=nn.Sequential(
            nn.Linear(32*32,128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,bottleneck_size)
        )
```

And the decoder.

```
self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32*32),
            nn.Sigmoid()
        )
```

Using these parts information can be encoded or decoded. After decoding the information should be almost identical to the information being encoded.

Loss is the difference between the original image and the decoded or reconstructed image.

```loss = torch.mean((images - reconstruction) ** 2)```






