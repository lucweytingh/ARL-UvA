###############################################################################
# MIT License
#
# Copyright (c) 2020 Jardenna Mohazzab, Luc Weytingh, 
#                    Casper Wortmann, Barbara Brocades Zaalberg
#
# This file contains the default parameters used for training the ARL model.
#
# Authors: Jardenna Mohazzab, Luc Weytingh, 
#          Casper Wortmann, Barbara Brocades Zaalberg 
# Date Created: 2021-01-01
###############################################################################

import argparse
import json

class DefaultArguments:
    def __init__(self):
        """
        Contains the default arguments passed to the main training function.

        Description of parameters:
            average_over: Number of iterations to average results over.
            dataset: Dataset to use: compas, uci_adult, or law_school.
            train_steps: Number of training steps.
            pretrain_steps: Number of steps to pretrain the learner.
            batch_size: Batch size to use for training.
            optimizer: Optimizer to use.
            embedding_size: The embedding size.
            lr_learner: Learning rate for the learner
            lr_adversary: Learning rate for the adversary.
            test_every: Evaluation interval.
            seed: The seed to use for reproducing the results.
            log_dir: Directory where the logs should be created.
            res_dir: Directory where the results should be created.
            print_loss: Print the loss and intermediate results.
        """
        self.average_over = 10
        self.dataset = "compas"
        self.train_steps = 1000
        self.pretrain_steps = 250
        self.batch_size = 32
        self.optimizer = "Adagrad"
        self.embedding_size = 32
        self.lr_learner = 0.01
        self.lr_adversary = 0.01
        self.test_every = 5
        self.seed = 42
        self.log_dir = "logs/"
        self.res_dir = "results/"
        self.print_loss = True
        self.model_name = "ARL"

    def update(self, new_args):
        """
        Change the class attributes given new arguments.

        Args:
            new_args: dict with {'attribute': value, [...]}.
        """
        for attr, value in new_args.items():
            setattr(self, attr, value)


def get_args():
    parser = argparse.ArgumentParser()
    default = DefaultArguments()

    parser.add_argument(
        "--average_over",
        default=5,
        type=int,
        help="Number of iterations to average results over",
    )

    # Model parameters
    parser.add_argument(
        "--model_name",
        default=default.model_name,
        type=str,
        help="Name of the model: ARL or baseline",
    )
    parser.add_argument(
        "--dataset",
        default=default.dataset,
        type=str,
        help="Dataset to use: uci_adult, compas, or law_school",
    )
    parser.add_argument(
        "--train_steps",
        default=default.train_steps,
        type=int,
        help="Number of training steps",
    )
    parser.add_argument(
        "--pretrain_steps",
        default=default.pretrain_steps,
        type=int,
        help="Number of steps to pretrain the learner",
    )
    parser.add_argument(
        "--batch_size",
        default=default.batch_size,
        type=int,
        help="Batch size to use for training",
    )
    parser.add_argument(
        "--optimizer", 
        default=default.optimizer, 
        type=str, 
        help="Optimizer to use"
    )
    parser.add_argument(
        "--embedding_size", 
        default=default.embedding_size, 
        type=int, 
        help="Embedding size"
    )
    parser.add_argument(
        "--lr_learner", 
        default=default.lr_learner, 
        type=float, 
        help="Learning rate for the learner"
    )
    parser.add_argument(
        "--lr_adversary",
        default=default.lr_adversary,
        type=float,
        help="Learning rate for the adversary",
    )
    parser.add_argument(
        "--test_every",
        default=default.test_every,
        type=int,
        help="Evaluation interval",
    )


    # Other hyperparameters
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seed to use for reproducing results",
    )
    parser.add_argument(
        "--log_dir",
        default="logs/",
        type=str,
        help="Directory where the logs should be created",
    )
    parser.add_argument(
        "--res_dir",
        default="results/",
        type=str,
        help="Directory where the results should be created",
    )
    parser.add_argument(
        "--print_loss",
        default=True,
        help="Print the loss and intermediate results",
    )

    return parser.parse_args()

def get_optimal_parameters(dataset, folder="hyperparameters/"):
    """
    Reads and returns the optimal hyperparameters generated with 
    the hyperparameter optimalisation.

    Args:
        dataset: Dataset to use: uci_adult, compas, or law_school.
        folder: The folder containing the hyperparameters.
    """
    param_file = folder + dataset + ".txt"
    with open(param_file) as f:
        params = json.loads(f.readlines()[0])

    return params