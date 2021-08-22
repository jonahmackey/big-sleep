import torch
from torch import nn
from collections import OrderedDict


class Alpha(nn.Module):
    def __init__(
        self,
        grid_range=1,
        size=512,
        num_layers=8,
        layer_width=24,
        order=2,
        pass_radius=False,
        add_dropout=False,
        p_dropout=0.5
    ):
      super().__init__()
      self.grid_range = grid_range
      self.size = size
      self.num_layers = num_layers
      self.layer_width = layer_width
      self.order = order
      self.pass_radius = pass_radius
      self.add_dropout = add_dropout
      self.p_dropout = p_dropout
      
      self.input_grid = self.make_grid()

      # TO-DO: make larger amount of hidden layers
      layers = []

      for i in range(num_layers):
        in_channels = layer_width
        out_channels = layer_width

        if i == 0:
          in_channels = 2 * order
          if pass_radius:
            in_channels += 1
        if i == num_layers - 1:
          out_channels = 1

        layers.append((f'conv{i}', nn.Conv2d(in_channels, out_channels, kernel_size=1)))

        if i < num_layers - 1:
          layers.append((f'relu{i}', nn.ReLU()))
          if add_dropout:
            layers.append((f'dropout{i}', nn.Dropout(p=p_dropout)))

      layers.append(('sigmoid', nn.Sigmoid()))

      self.network = nn.Sequential(OrderedDict(layers))

    def forward(self):
      mask = self.network(self.input_grid)

      return mask

    def make_grid(self):
      coord_range = torch.linspace(-self.grid_range, self.grid_range, self.size)
      x = coord_range.view(-1, 1).repeat(1, coord_range.size(0))
      y = coord_range.view(1, -1).repeat(coord_range.size(0), 1)

      grid = torch.stack([x, y], dim=0).unsqueeze(0)

      if self.pass_radius:
        r = x**2 + y**2
        r = torch.unsqueeze(torch.unsqueeze(r, 0), 0)
        r = torch.sqrt(r)
        grid = torch.cat([grid, r], dim=1)

      # concatenate higher power grids
      for p in range(2, self.order + 1):
        x2 = x**p
        y2 = y**p

        grid2 = torch.stack([x2, y2], dim=0).unsqueeze(0)
        grid = torch.cat([grid, grid2], dim=1)

      return nn.parameter.Parameter(grid, requires_grad=False)
