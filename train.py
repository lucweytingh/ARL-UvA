###############################################################################
# MIT License
#
# Copyright (c) 2020 Jardenna Mohazzab, Luc Weytingh, 
#                    Casper Wortmann, Barbara Brocades Zaalberg
#
# This file contains the functions used for training the ARL model.
#
# Authors: Jardenna Mohazzab, Luc Weytingh, 
#          Casper Wortmann, Barbara Brocades Zaalberg 
# Date Created: 2021-01-01
###############################################################################

import os
import datetime
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from dataloader import *
from arl import ARL
from baseline import baseline
from metrics import FairnessMetrics
from argparser import get_args

def train_model(
        model,
        train_loader,
        test_dataset,
        train_steps,
        test_every,
        pretrain_steps,
        optimizer_learner,
        optimizer_adv,
        metrics,
        checkpoint_dir,
        logger_learner,
        logger_adv,
        logger_metrics,
        n_iters,
        print_loss=True,
        device="cpu",
    ):
    """
    Function for training the ARL model on a dataset for a single epoch.

    Args:
        model: ARL model to train.
        train_loader: Data Loader for the dataset you want to train on.
        test_dataset: Data iterator of the test set. 
        train_steps: Number of training steps. 
        test_every: Evaluation interval. 
        pretrain_steps: Number of pretrain steps (no adversary training).
        optimizer_learner: The optimizer used to update the learner.
        optimizer_adv: The optimizer used to update the adversary.
        metrics: Metrics objects for saving AUC (see metrics.py).
        checkpoint_dir: Directory to save the tensorboard checkpoints. 
        logger_learner: Object in which loss learner curves are stored.
        logger_adv: Object in which loss learner curves are stored (adversary).
        logger_metrics: Objects in which AUC metrics are stored. 
        n_iters: How often to train the model with different seeds.
        print_loss: Wheter to print the loss values. 
        device: device to train on (cpu / gpu),
    """
    test_cat, test_num, test_target = test_dataset[:]
    model.train()
    loss_adv = 0
    train = True
    total_steps = 0

    # Reset the dataloader if out of data.
    while train:
        for step, (train_cat, train_num, train_target) in enumerate(
            train_loader
        ):
            # Transfer data to GPU if possible. 
            train_cat = train_cat.to(device)
            train_num = train_num.to(device)
            train_target = train_target.to(device)
            total_steps += 1

            # Learner update step.
            loss_learner, train_logits, logging_dict = model.learner_step(
                train_cat, train_num, train_target
            )
            logger_learner.add_values(logging_dict)
            optimizer_learner.step()

            # Adversary update step (if ARL model).
            if optimizer_adv:
                if total_steps >= pretrain_steps:
                    loss_adv, logging_dict = model.adversary_step(
                        train_cat, train_num, train_logits, train_target
                    )
                    logger_adv.add_values(logging_dict)
                    optimizer_adv.step()
                else:
                    loss_adv = -loss_learner
                    logger_adv.add_values({"adv_loss": torch.Tensor([loss_adv])})

            # Evaluate on test set.
            if total_steps % test_every == 0:
                test_cat = test_cat.to(device)
                test_num = test_num.to(device)

                with torch.no_grad():
                    test_logits, test_sigmoid, test_pred = model.learner(
                        test_cat, test_num
                    )

                # Calculate AUC and accuracy metrics. 
                auc = metrics.set_auc(
                    test_sigmoid, test_target, n_iters
                )
                auc_min, auc_macro_avg, auc_minority = metrics.set_auc_other(test_sigmoid, test_target, n_iters, test_dataset)
                acc = metrics.set_acc(test_pred, test_target, n_iters)
                metrics.set_posnegs(test_pred, test_target, n_iters)

                # Add the metrics to tensorboard.
                logger_metrics.add_values(metrics.logging_dict)

                if print_loss:
                    print(
                        "--- {} loss learner: {:.3f} auc: {:.3f} acc: {:.3f}".format(
                            total_steps, loss_learner, auc, acc
                        )
                    )

            # Save the most recent model.
            if (total_steps + 1) % 10 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(checkpoint_dir, "model_checkpoint.pt"),
                )

            # Stop if max training steps reached.
            if (total_steps + 1) > train_steps:
                if print_loss:
                    print("Max training steps reached")
                train = False
                break


def train_for_n_iters(
        train_dataset,
        test_dataset,
        model_params,
        lr_params,
        n_iters=5,
        train_steps=1000,
        test_every=10,
        pretrain_steps=250,
        print_loss=True,
        log_dir="logs/",
        model_name="ARL"
    ):
    """
    Trains the ARL model for n iterations, and averages the results. 

    Args:
        train_dataset: Data iterator of the train set.
        test_dataset: Data iterator of the test set. 
        model_params: A dictionary with model hyperparameters. 
        lr_params: A dictionary with hyperparmaeters for optimizers.
        n_iters: How often to train the model with different seeds.
        train_steps: Number of training steps. 
        test_every: How often to evaluate on test set. 
        pretrain_steps: Number of pretrain steps (steps with no adversary).
        print_loss: Wheter to print the loss during training. 
        log_dir: Directory where to save the tensorboard loggers. 
    """
    # Set the device on which to train. 
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_params["device"] = device

    # Initiate metrics object.
    metrics = FairnessMetrics(n_iters, test_every)

    # Preparation of logging directories.
    experiment_dir = os.path.join(
        log_dir, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    )
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialte TensorBoard loggers.
    summary_writer = SummaryWriter(experiment_dir)
    logger_learner = TensorBoardLogger(summary_writer, name="learner")
    logger_adv = TensorBoardLogger(summary_writer, name="adversary")
    logger_metrics = TensorBoardLogger(summary_writer, name="metrics")

    for i in range(n_iters):
        print(f"Training model {i + 1}/{n_iters}")
        seed_everything(42 + i)

        # Load the train dataset as a pytorch dataloader.
        train_loader = DataLoader(
            train_dataset, batch_size=model_params["batch_size"], shuffle=True
        )

        # Create the model.
        if model_name == "ARL":
            model = ARL(**model_params)
        elif model_name == "baseline":
            model = baseline(**model_params)
        else:
            print("Unknown model")

        # Transfer model to correct device.
        model = model.to(device)

        # Adagrad is the defeault optimizer.
        optimizer_learner = torch.optim.Adagrad(
            model.learner.parameters(), lr=lr_params["learner"]
        )
        if model_name == 'ARL':
            optimizer_adv = torch.optim.Adagrad(
                model.adversary.parameters(), lr=lr_params["adversary"]
            )
        elif model_name == 'baseline':
            optimizer_adv = None

        # Train the model with current seeds.
        if print_loss:
            print("Start training on device {}".format(device))
        train_model(
            model,
            train_loader,
            test_dataset,
            train_steps,
            test_every,
            pretrain_steps,
            optimizer_learner,
            optimizer_adv,
            metrics,
            checkpoint_dir,
            logger_learner,
            logger_adv,
            logger_metrics,
            n_iters=i,
            print_loss=print_loss,
            device=device,
        )

    # Average results and return metrics
    metrics.average_results()
    return metrics


def seed_everything(seed):
    """
    Changes the seed for reproducibility. 
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    """
    Main Function for the full training loop.

    Inputs:
        args: Namespace object from the argument parser.
    """
    # Load the train and test sets
    train_dataset = loadDataset(
        dataset=args.dataset,
        train_or_test="train",
        embedding_size=args.embedding_size,
    )
    test_dataset = loadDataset(dataset=args.dataset, train_or_test="test")

    # Set the model parameters. 
    model_params = {}
    model_params["learner_hidden_units"] = [64, 32]
    model_params["batch_size"] = args.batch_size
    model_params["embedding_size"] = train_dataset.categorical_embedding_sizes
    model_params["n_num_cols"] = len(train_dataset.mean_std.keys())
    if args.model_name == "ARL":
        model_params["adversary_hidden_units"] = [32]

    # Set the parameters of the optimizers.
    lr_params = {}
    lr_params["learner"] = args.lr_learner
    lr_params["adversary"] = args.lr_adversary

    # Calculate the average results when training over N iterations.
    metrics = train_for_n_iters(
        train_dataset,
        test_dataset,
        model_params,
        lr_params,
        args.average_over,
        args.train_steps,
        args.test_every,
        args.pretrain_steps,
        log_dir=args.log_dir,
        print_loss=args.print_loss,
        model_name=args.model_name,
    )

    # Save the metrics to output file.
    os.makedirs(args.res_dir, exist_ok=True)
    metrics.save_metrics(args.res_dir, args.dataset, args.model_name)

    print("Done training\n")
    print("-" * 35)
    print("Results\n")
    print("Average AUC: {:.3f} \u00B1 {:.4f}".format(metrics.auc_avg[-1], metrics.auc_std[-1]))
    print("Average AUC(macro-avg): {:.3f}".format(metrics.auc_macro_avg[-1]))
    print("Average AUC(min): {:.3f}".format(metrics.auc_min_avg[-1]))
    print("Average AUC(minority): {:.3f}".format(metrics.auc_minority_avg[-1]))
    print("-" * 35 + "\n")


if __name__ == "__main__":
    # Get the default and command line arguments.
    args = get_args()

    # Run the model.
    main(args)
