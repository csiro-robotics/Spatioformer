import torch.nn as nn


class CNNModel(nn.Module):   
    def __init__(self, dropout=0.1, num_bands=6, num_filters=8, nodes_hidden=1024):
        super(CNNModel, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(num_bands, num_filters, 3),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters),
            nn.Conv2d(num_filters, num_filters, 3),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters),
            nn.Conv2d(num_filters, num_filters, 3),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters),
            nn.Dropout(dropout),
        )

        self.feedforward = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters * 9, nodes_hidden),
            nn.Linear(nodes_hidden, 1),
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.feedforward(x)
        return x

