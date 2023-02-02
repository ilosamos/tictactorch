import torch
from torch import nn


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.filters = 16
        self.kernel = 3
        self.padding = 2
        self.width = self.kernel + self.padding # conv converts 3x3 board to 5x5 with depth of above filter size

        # using convolution since we have a nice 3x3 board, probably overkill tho
        self.conv = nn.Conv2d(in_channels=1, out_channels=self.filters, kernel_size=self.kernel, stride=1, padding=self.padding, bias=True)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(self.filters * self.width * self.width)
        self.linear = nn.Linear(self.filters * self.width * self.width, 9)
        self.softmax = nn.Softmax(dim=1)

        total_params = sum(p.numel() for p in self.parameters())
        print("Number of parameters:", total_params)

    def forward(self, x):
        x = x.to(torch.float32)
        x = x.unsqueeze(1) # add channel dimension
        x = self.conv(x)
        x = self.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.norm(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x

    @torch.no_grad()
    def sample(self, x):
        x = x.unsqueeze(0) # add batch dimension
        x = self.forward(x)
        # x = torch.multinomial(x, 1)
        x = torch.argmax(x)
        return x.item()

    def load_from_file(self, path):
        checkpoint = torch.load(path)
        print("Loading checkpoint, AVG reward from checkpoint is:", checkpoint['avg_reward'])
        self.load_state_dict(checkpoint['model'])
