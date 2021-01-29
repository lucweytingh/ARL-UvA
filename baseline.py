###############################################################################
# MIT License
#
# Copyright (c) 2020 Jardenna Mohazzab, Luc Weytingh, 
#                    Casper Wortmann, Barbara Brocades Zaalberg
#
# This file contains an implementation of a baseline NN model.
#
# Author: Jardenna Mohazzab, Luc Weytingh, 
#         Casper Wortmann, Barbara Brocades Zaalberg 
# Date Created: 2021-01-01
###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class baselineNN(nn.Module):
    def __init__(
        self, embedding_size, n_num_cols, n_hidden, 
        activation_fn=nn.ReLU, device='cpu'
    ):
        """
        Implements the learner DNN.
        Args:
          embedding_size: list of tuples (n_classes, n_features) containing
                           embedding sizes for categorical columns.
          n_num_cols: number of numerical inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer.
          activation_fn: the activation function to use.
        """
        super().__init__()
        self.device = device

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(n_classes, n_features)
                for n_classes, n_features in embedding_size
            ]
        )

        all_layers = []
        n_cat_cols = sum((n_features for _, n_features in embedding_size))
        input_size = n_cat_cols + n_num_cols

        for dim in n_hidden:
            all_layers.append(nn.Linear(input_size, dim))
            all_layers.append(activation_fn())
            input_size = dim

        all_layers.append(nn.Linear(n_hidden[-1], 1))

        self.layers = nn.Sequential(*all_layers)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x_cat, x_num):
        """
        The forward step.
        """

        embedding_cols = []
        for i, emb in enumerate(self.embeddings):
            embedding_cols.append(emb(x_cat[:, i]))

        x = torch.cat(embedding_cols, dim=1)
        x = torch.cat([x, x_num], dim=1)

        # Get the logits output (for calculating loss)
        logits = self.layers(x)

        sigmoid_output = self.sigmoid(logits)

        sigmoid_output.to(self.device)
        class_predictions = torch.where(
            sigmoid_output > 0.5,
            torch.tensor(1, dtype=torch.float32).to(self.device),
            torch.tensor(0, dtype=torch.float32).to(self.device),
        )

        return logits, sigmoid_output, class_predictions



class baseline(nn.Module):

    def __init__(
        self,
        embedding_size,
        n_num_cols,
        learner_hidden_units=[64, 32],
        batch_size=256,
        activation_fn=nn.ReLU,
        device='cpu',
    ):
        """
        Combines the Learner and Adversary into a single module.

        Args:
          embedding_size: list of tuples (n_classes, embedding_dim) containing
                    embedding sizes for categorical columns.
          n_num_cols: the amount of numerical columns in the data.
          learner_hidden_units: list of ints, specifies the number of units
                    in each linear layer for the learner.
          adversary_hidden_units: list of ints, specifies the number of units
                    in each linear layer for the learner.
          batch_size: the batch size.
          activation_fn: the activation function to use for the learner.
        """
        super().__init__()
        torch.autograd.set_detect_anomaly(True)

        self.device = device

        self.learner = baselineNN(
            embedding_size,
            n_num_cols,
            learner_hidden_units,
            activation_fn=activation_fn,
            device=device
        )

        self.learner.to(device)

    def learner_step(self, x_cat, x_num, targets):
        self.learner.zero_grad()
        logits, _, _ = self.learner(x_cat, x_num)
        loss = torch.mean(F.binary_cross_entropy_with_logits(logits, targets))
        loss.backward()

        # Predictions are returned to trainer for fairness metrics
        logging_dict = {"learner_loss": loss}
        return loss, logits, logging_dict


    def get_learner_loss(self, logits, targets, adversary_weights):
        """
        Compute the loss.
        """
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss = torch.mean(loss)
        return loss
