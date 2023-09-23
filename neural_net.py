import torch.nn as nn


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        """
        Define the layers of the network.
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        """
        Define specify how data will pass through the network.
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits