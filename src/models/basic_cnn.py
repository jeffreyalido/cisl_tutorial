import torch.nn as nn

class DenoisingCNN(nn.Module):
    def __init__(self, config):
        super(DenoisingCNN, self).__init__()
        num_encoder_layers = config.get('num_encoder_layers', 3)
        num_decoder_layers = config.get('num_decoder_layers', 3)

        # Build the encoder and decoder using the specified number of layers
        self.encoder = self.build_encoder(num_encoder_layers)
        self.decoder = self.build_decoder(num_decoder_layers)

    def build_encoder(self, num_layers):
        layers = []
        in_channels = 3
        for i in range(num_layers):
            out_channels = 64 if i == 0 else 32
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        return nn.Sequential(*layers)

    def build_decoder(self, num_layers):
        layers = []
        in_channels = 32
        for i in range(num_layers):
            out_channels = 3 if i == num_layers - 1 else 32
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        # Replace the final ReLU in decoder with a linear activation (output layer)
        layers.pop()  # Remove the last ReLU for the output layer
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
