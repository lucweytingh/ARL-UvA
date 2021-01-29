###############################################################################
# MIT License
#
# Copyright (c) 2020 Jardenna Mohazzab, Luc Weytingh, 
#                    Casper Wortmann, Barbara Brocades Zaalberg
#
# This file saves ARL and baseline results (save_all_results function), and
# performs a significance test on these results (significance test function). 
# To run: python significance.py
#
# Author: Jardenna Mohazzab, Luc Weytingh, 
#         Casper Wortmann, Barbara Brocades Zaalberg 
# Date Created: 2021-01-01
###############################################################################

import json

import train
from dataloader import *
from arl import ARL
from metrics import SignificanceMetrics
from argparser import get_args, get_optimal_parameters

def save_all_results(args, optimal_params):
    """
    Save all results for all datasets, with the optimum parameters.

    Inputs:
        args: Namespace object from the argument parser.
        optimal_params: a dict with the optimum parameters for each dataset.
    """
    # Iterate over all possible datasets and change the parser.
    for dataset in ['compas', 'uci_adult', 'law_school']:
        setattr(args, 'dataset', dataset)
        setattr(args, 'batch_size', optimal_params[dataset]['batch_size'])
        setattr(args, 'lr_learner', optimal_params[dataset]['lr_learner'])
        setattr(args, 'lr_adversary', optimal_params[dataset]['lr_adversary'])
        setattr(args, 'train_steps', optimal_params[dataset]['train_steps'])

        # Train the ARL model and the baseline model.
        for model in ["ARL", "baseline"]:
            setattr(args, 'model_name', model)
            print("train {} with {}".format(
                args.dataset, args.model_name))
            train.main(args)


def test_significance(dataset, res_dir="results/"):
    """
    Tests the significance of the ARL model for a given dataset. 

    Args:
        res_dir: the directory the results were saved in.
    """

    metrics = SignificanceMetrics()
    # Open up json with results for baseline and ARL.
    with open(res_dir + dataset + '_baseline.json') as f:
        base = json.load(f)
    with open(res_dir + dataset + '_ARL.json') as f:
        arl = json.load(f)

    pvalue = metrics.calc_pval(base['tp'], base['tn'], 
        arl['tp'], arl['tn'], base['auc_avg_final'], arl['auc_avg_final'])

    # Determine if p-value is significant. 
    print("The p-value for {} is {:.3f}".format(dataset, pvalue))
    if pvalue < 0.05:
        print("This difference is significant.")
    else:
        print("This difference is not significant")

if __name__ == "__main__":
    # Get the default and command line arguments.
    args = get_args()
    
    # Never print the loss while doing signtest.
    setattr(args, 'print_loss', False)

    # Run the significance test as an average over 10 runs.
    setattr(args, 'average_over', 10)

    # Optimal hyperparameters.
    optimal_pars = {
        "compas": get_optimal_parameters("compas"),
        "uci_adult": get_optimal_parameters("uci_adult"),
        "law_school": get_optimal_parameters("law_school")
    }

    # Save results for training baseline and ARL model. 
    save_all_results(args, optimal_pars)

    # Perform a significance test for each models result.
    for dataset in ['compas', 'uci_adult', 'law_school']:
        test_significance(dataset, args.res_dir)

