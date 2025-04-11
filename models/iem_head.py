import torch
import torch.nn as nn

class serverModel(nn.Module):

    def __init__(self, num_classes=6, num_clients=3, dim=256):
        super(serverModel, self).__init__()
        self.fc = nn.Linear(dim * num_clients, num_classes)

    def forward(self, x):
        pooled_view = self.fc(x)
        return pooled_view
