import torch.nn as nn

class AEncoder(nn.Module):
    def __init__(self, bottleneck_size=2):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Linear(32*32,128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,bottleneck_size)
        )
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
    def forward_enc(self,x):
        return self.encoder(x)
    def forward_dec(self,x):
        x= self.decoder(x)
        x=x.reshape(-1,1,32,32)
        return x

    def forward(self,x):
        batch, channels, height, width = x.shape
        x=x.flatten(1)
        print(x.shape)
        enc=self.forward_enc(x)
        dec=self.forward_dec(enc)
        print(dec.shape)
        return enc,dec


class SimpleAEncoder(nn.Module):
    def __init__(self, bottleneck_size=2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(32 * 32, bottleneck_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, 32 * 32),
        )

    def forward(self, x):
        batch, channels, height, width = x.shape

        ### Flatten Image to Vector ###
        x = x.flatten(1)

        ### Pass Through Encoder ###
        enc = self.encoder(x)

        ### Pass Through Decoder ###
        dec = self.decoder(enc)

        ### Put Decoded Image Back to Original Shape ###
        dec = dec.reshape(batch, channels, height, width)

        return enc, dec


class ConvolutionalAEncoder(nn.Module):
    def __init__(self, in_channels=1, channels_bottleneck=4):
        super().__init__()
        self.bottleneck = channels_bottleneck
        self.in_channels = in_channels

        self.encoder_conv = nn.Sequential(

            nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=self.bottleneck, kernel_size=3, stride=2, padding=1),

        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.bottleneck, out_channels=16, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=8, out_channels=in_channels, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.Sigmoid()

        )

    def forward_enc(self, x):
        conv_enc = self.encoder_conv(x)
        return conv_enc

    def forward_dec(self, x):
        return self.decoder_conv(x)

    def forward(self, x):
        batch_size, num_channels, height, width = x.shape
        enc = self.forward_enc(x)
        dec = self.forward_dec(enc)

        return enc, dec