from argparse import Namespace

import torch
import torch.nn as nn

from .mpn import MPN
from chemprop.nn_utils import get_activation_function, initialize_weights


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, classification: bool, multiclass: bool):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        """
        super(MoleculeModel, self).__init__()

        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)

    def create_encoder(self, args: Namespace):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args)

    def _create_ffn(self, first_linear_dim: int, ffn_hidden_size: int, ffn_num_layers: int,
                    output_size: int, dropout: nn.Module, activation) -> nn.Sequential:
        """
        Create FFN layers
        :param dropout:
        :param args:
        :return:
        """
        if ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, ffn_hidden_size)
            ]
            for _ in range(ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(ffn_hidden_size, ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(ffn_hidden_size, output_size),
            ])

            return nn.Sequential(*ffn)

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size
            if args.use_input_features:
                first_linear_dim += args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN model
        self.ffn = []
        self.constraints = args.constraints if args.constraints is not None else None
        for i, target in enumerate(args.targets):
            ffn_i = {}
            ffn_i['ffn'] = self._create_ffn(first_linear_dim, args.ffn_hidden_size,
                                            args.ffn_num_layers, args.output_size, dropout, activation)
            if self.constraints is not None and i < len(self.constraints):
                ffn_i['weights_readout'] = self._create_ffn(first_linear_dim, args.ffn_hidden_size,
                                                            args.ffn_num_layers, args.output_size,
                                                            dropout, activation)
                ffn_i['constraint'] = self.constraints[i]

            self.ffn.append(ffn_i)

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        hidden, a_scope = self.encoder(*input)

        output_all = []
        for ffn in self.ffn:
            output = ffn['ffn'](hidden)
            constrained_output = []
            if 'constraint' in ffn:
                weights = ffn['weights_readout'](hidden)
                for i, (a_start, a_size) in enumerate(a_scope):
                    if a_size == 0:
                        continue
                    else:
                        cur_weights = weights.narrow(0, a_start, a_size)
                        cur_output = output.narrow(0, a_start, a_size)

                        cur_weights_sum = cur_weights.sum()
                        cur_output_sum = cur_output.sum()

                        cur_output = cur_output + cur_weights * \
                                     (ffn['constraint'] - cur_output_sum) / cur_weights_sum
                        constrained_output.append(cur_output)
                cur_output = torch.cat(constrained_output, dim=0)
            else:
                cur_output = output[1:]
            output_all.append(cur_output)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape(
                (output.size(0), -1, self.num_classes))  # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(
                    output)  # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output_all

def build_model(args: Namespace) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    # FIXME
    args.output_size = 1
    if args.dataset_type == 'multiclass':
        args.output_size *= args.multiclass_num_classes

    model = MoleculeModel(classification=args.dataset_type == 'classification',
                          multiclass=args.dataset_type == 'multiclass')
    model.create_encoder(args)
    model.create_ffn(args)

    initialize_weights(model)

    return model
