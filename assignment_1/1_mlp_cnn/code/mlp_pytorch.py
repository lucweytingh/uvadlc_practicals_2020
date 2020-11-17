"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """
        super().__init__()

        self.all_modules = nn.ModuleList()

        for i in range(len(n_hidden) - 1):
            self.all_modules.append(nn.Linear(n_hidden[i], n_hidden[i + 1]))
            self.all_modules.append(nn.ELU())

        if len(n_hidden) > 0:
            self.all_modules.insert(0, nn.Linear(n_inputs, n_hidden[0]))
            self.all_modules.insert(1, nn.ELU())
            self.all_modules.append(nn.Linear(n_hidden[-1], n_classes))
        else:
            self.all_modules.insert(0, nn.Linear(n_inputs, n_classes))

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """
        # out = self.all_modules(x)

        out = self.all_modules[0](x)
        for module in self.all_modules[1:]:
            out = module(out)

        return out
