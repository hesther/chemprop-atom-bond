from argparse import Namespace
from typing import List, Union

import torch
import torch.nn as nn
import re
import numpy as np

from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import index_select_ND, get_activation_function

class AttrProxy(object):
    """Translates index lookups into attribute lookups"""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __len__(self):
        return len([x for x in self.module.__dict__['_modules'].keys() if re.match(f'{self.prefix}\d+', x)])

    def __getitem__(self, item):
        if item >= len(self):
            raise IndexError
        return getattr(self.module, self.prefix + str(item))

class MultiReadout(nn.Module):
    """A fake list of FFNs for reading out as suggested in
    https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/3 """

    def __init__(self, args: Namespace, targets, constraints=None):
        """

        :param args:
        :param args:
        :param constraints:
        """

        features_size = args.hidden_size
        hidden_size = args.ffn_hidden_size
        num_layers = args.ffn_num_layers
        output_size = args.output_size
        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        super(MultiReadout, self).__init__()
        for i, target in enumerate(targets):
            constraint = constraints[i] if i < len(constraints) else None
            self.add_module(f'readout_{i}', FFN(features_size, hidden_size, num_layers,
                                                output_size, dropout, activation, constraint))

        self.ffn_list = AttrProxy(self, 'readout_')

    def forward(self, *input):
        return [ffn(*input) for ffn in self.ffn_list]


class FFN(nn.Module):
    """A Feedforward netowrk reading out properties from fingerprint"""

    def __init__(self, features_size, hidden_size, num_layers, output_size, dropout, activation, constraint=None):
        """Initializes the FFN.

        args: Arguments.
        constraints: constraints applied to output
        """

        super(FFN, self).__init__()
        self.ffn = DenseLayers(features_size, hidden_size,
                               num_layers, output_size, dropout, activation)

        if constraint is not None:
            self.weights_readout = DenseLayers(features_size, hidden_size,
                                               num_layers, output_size, dropout, activation)
            self.constraint = constraint
        else:
            self.constraint = None

    def forward(self, input):
        """
        Runs the FFN on input

        :param input:
        :return:
        """
        hidden, a_scope = input

        output = self.ffn(hidden)
        if self.constraint is not None:
            weights = self.weights_readout(hidden)
            constrained_output = []
            for i, (a_start, a_size) in enumerate(a_scope):
                if a_size == 0:
                    continue
                else:
                    cur_weights = weights.narrow(0, a_start, a_size)
                    cur_output = output.narrow(0, a_start, a_size)

                    cur_weights_sum = cur_weights.sum()
                    cur_output_sum = cur_output.sum()

                    cur_output = cur_output + cur_weights * \
                                 (self.constraint - cur_output_sum) / cur_weights_sum
                    constrained_output.append(cur_output)
            output = torch.cat(constrained_output, dim=0)
        else:
            output = output[1:]

        return output


class DenseLayers(nn.Module):
    "Dense layers"

    def __init__(self,
                 first_linear_dim: int,
                 hidden_size: int,
                 num_layers: int,
                 output_size: int,
                 dropout: nn.Module,
                 activation) -> nn.Sequential:
        """
        :param first_linear_dim:
        :param hidden_size:
        :param num_layers:
        :param output_size:
        :param dropout:
        :param activation:
        """
        super(DenseLayers, self).__init__()
        if num_layers == 1:
            layers = [
                dropout,
                nn.Linear(first_linear_dim, output_size)
            ]
        else:
            layers = [
                dropout,
                nn.Linear(first_linear_dim, hidden_size)
            ]
            for _ in range(num_layers - 2):
                layers.extend([
                    activation,
                    dropout,
                    nn.Linear(hidden_size, hidden_size),
                ])
            layers.extend([
                activation,
                dropout,
                nn.Linear(hidden_size, output_size),
            ])

        self.dense_layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.dense_layers(input)