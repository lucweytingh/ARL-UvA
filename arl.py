###############################################################################
# MIT License
#
# Copyright (c) 2020 Jardenna Mohazzab, Luc Weytingh, 
#                    Casper Wortmann, Barbara Brocades Zaalberg
#
# This file contains an implementation of the ARL model prented in "Fairness 
# without Demographics through Adversarially Reweighted Learning" by Lahoti 
# et al..
#
# Author: Jardenna Mohazzab, Luc Weytingh, 
#         Casper Wortmann, Barbara Brocades Zaalberg 
# Date Created: 2021-01-01
###############################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LearnerNN(nn.Module):
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
        The forward step for the learner.
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


class AdversaryNN(nn.Module):
    def __init__(self, embedding_size, n_num_cols, n_hidden, device='cpu'):
        """
        Implements the adversary DNN.
        Args:
          embedding_size: list of tuples (n_classes, n_features) containing
                          embedding sizes for categorical columns.
          n_num_cols: number of numerical inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer.
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
            input_size = dim

        all_layers.append(nn.Linear(n_hidden[-1], 1))
        # all_layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*all_layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_cat, x_num):
        """
        The forward step for the adversary.
        """
        embedding_cols = []
        for i, emb in enumerate(self.embeddings):
            embedding_cols.append(emb(x_cat[:, i]))

        x = torch.cat(embedding_cols, dim=1)

        x = torch.cat([x, x_num], dim=1)
        x = self.layers(x)

        x = self.sigmoid(x)
        x_mean = torch.mean(x)
        x = x / torch.max(torch.Tensor([x_mean, 1e-4]))
        x = x + torch.ones_like(x)

        return x


class ARL(nn.Module):

    def __init__(
        self,
        embedding_size,
        n_num_cols,
        learner_hidden_units=[64, 32],
        adversary_hidden_units=[32],
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
        self.adversary_weights = torch.ones(batch_size, 1)

        self.learner = LearnerNN(
            embedding_size,
            n_num_cols,
            learner_hidden_units,
            activation_fn=activation_fn,
            device=device
        )
        self.adversary = AdversaryNN(
            embedding_size, n_num_cols, adversary_hidden_units, device=device
        )

        self.learner.to(device)
        self.adversary.to(device)


    def learner_step(self, x_cat, x_num, targets):
        self.learner.zero_grad()
        logits, _, _ = self.learner(x_cat, x_num)

        # Logits are used for calculating loss
        adversary_weights = self.adversary_weights.to(self.device)
        loss = self.get_learner_loss(logits, targets, adversary_weights)
        loss.backward()

        # Predictions are returned to trainer for fairness metrics
        logging_dict = {"learner_loss": loss}
        return loss, logits, logging_dict

    def adversary_step(self, x_cat, x_num, learner_logits, targets):
        """
        Performs one loop
        """
        self.adversary.zero_grad()

        adversary_weights = self.adversary(x_cat, x_num)
        self.adversary_weights = adversary_weights.detach()

        loss = self.get_adversary_loss(
            learner_logits.detach(), targets, adversary_weights
        )

        loss.backward()

        logging_dict = {"adv_loss": loss}
        return loss, logging_dict

    def get_learner_loss(self, logits, targets, adversary_weights):
        """
        Compute the loss for the learner.
        """
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss.to(self.device)
        
        weighted_loss = loss * adversary_weights
        weighted_loss = torch.mean(weighted_loss)
        return weighted_loss

    def get_adversary_loss(self, logits, targets, adversary_weights):
        """
        Compute the loss for the adversary.
        """
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss.to(self.device)

        weighted_loss = -(adversary_weights * loss)
        weighted_loss = torch.mean(weighted_loss)
        return weighted_loss
