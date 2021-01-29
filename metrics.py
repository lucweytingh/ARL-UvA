###############################################################################
# MIT License
#
# Copyright (c) 2020 Jardenna Mohazzab, Luc Weytingh, 
#                    Casper Wortmann, Barbara Brocades Zaalberg
#
# This file contains the functions used for evaluating the ARL model 
# (FairnessMetrics), and testing the significance of the ARL model 
# (SignificanceMetrics).
#
# Author: Jardenna Mohazzab, Luc Weytingh, 
#         Casper Wortmann, Barbara Brocades Zaalberg 
# Date Created: 2021-01-01
###############################################################################

import torch
import os
import numpy as np
import json
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from scipy.stats import norm

class FairnessMetrics():
    def __init__(self, averaged_over, eval_every=5):
        """
        Implements evaluation metrics for measuring performance of the 
        ARL model.

        Args:
            averaged_over: the amount of iterations the model is ran (for averaging 
                     the results).
            eval_every: the amount of steps between evaluation of the model.
        """
        self.logging_dict = {"auc": 0,
                            "acc": 0}
        self.eval_every = eval_every
        self.auc = [[] for i in range(averaged_over)]
        self.auc_min = [[] for i in range(averaged_over)]
        self.auc_max = [[] for i in range(averaged_over)]
        self.auc_macro = [[] for i in range(averaged_over)]
        self.auc_minority = [[] for i in range(averaged_over)]

        self.acc = [[] for i in range(averaged_over)]

        # Save the achieved target metrics (tn, fp, fn, tp)
        self.posnegs = [(0,0,0,0) for i in range(averaged_over)]

    def calc_auc(self, pred, targets):
        """
        Calculates the AUC score given the predictions and targets.

        Args:
            pred: prediction (Torch tensor).
            targets: target varialbles (Torch tensor).
        """
        try:
            auc = roc_auc_score(targets.cpu().detach().numpy(), pred.cpu().detach().numpy())

        except ValueError:
            # AUC can only be calculated if there are FP, FN, TP and TN
            # TODO: How does tensorflow handle this?
            print("Your data is too unbalanced to calculate AUC")
            auc = 0
        return auc

    def set_auc(self, pred, targets, n_iter):
        """
        Calculates AUC: ROC area under the curve.

        Args:
            pred: prediction (Torch tensor).
            targets: target varialbles (Torch tensor).
            n_iter: iteration of this training loop.
        """
        auc = self.calc_auc(pred, targets)

        self.auc[n_iter].append(auc)
        self.logging_dict["auc"] = auc
        return auc

    def set_auc_other(self, pred, targets, n_iter, dataset):
        """
        Calculates AUC(min): minimum AUC over all protected groups.
        Calculates AUC(maximum): maximum AUC over all protected groups.
        Calculates AUC (minority): the AUC reported for the smallest protected 
                                   group in the dataset.

        Args:
            pred: prediction (Torch tensor).
            targets: target varialbles (Torch tensor).
            n_iter: iteration of this training loop.
        """
        aucs = []
        for group in dataset.subgroup_indexes:
            pred_group = pred[group]
            targets_group = targets[group]
            aucs.append(self.calc_auc(pred_group, targets_group))

        auc_min = min(aucs)
        auc_max = max(aucs)
        auc_macro = np.mean(aucs)
        auc_minority = aucs[dataset.subgroup_minority]
        
        self.auc_min[n_iter].append(auc_min)
        self.auc_max[n_iter].append(auc_max)
        self.auc_macro[n_iter].append(auc_macro)
        self.auc_minority[n_iter].append(auc_minority)

        self.logging_dict["auc_min"] = auc_min
        self.logging_dict["auc_max"] = auc_max
        self.logging_dict["auc_minority"] = auc_minority
        self.logging_dict["auc_macro"] = auc_macro
        return auc_min, auc_macro, auc_minority

    def set_acc(self, pred, targets, n_iter):
        """
        Calculates the accuracy score.

        Args:
            pred: prediction (Torch tensor).
            targets: target varialbles (Torch tensor).
            n_iter: iteration of this training loop. 
        """

        acc = accuracy_score(targets.cpu().detach().numpy(), pred.cpu().detach().numpy())
        self.acc[n_iter].append(acc)
        self.logging_dict["acc"] = acc
        
        return acc

    def set_posnegs(self, pred, targets, n_iter):
        """
        Calculates true positives, true negatives, 
        false positives, false negatives.

        Args:
            pred: prediction (Torch tensor).
            targets: target varialbles (Torch tensor).
            n_iter: iteration of this training loop.            
        """
        tn, fp, fn, tp = confusion_matrix(targets.cpu().detach().numpy(), pred.cpu().detach().numpy()).ravel()
        self.posnegs[n_iter] = np.array([tn, fp, fn, tp])
        

    def average_results(self):
        """
        Averages the results of all iterations.
        """
        self.auc_avg = np.mean(np.array(self.auc), axis=0)
        self.auc_min_avg = np.mean(np.array(self.auc_min), axis=0)
        self.auc_max_avg = np.mean(np.array(self.auc_max), axis=0)
        self.auc_macro_avg = np.mean(np.array(self.auc_macro), axis=0)
        self.auc_minority_avg = np.mean(np.array(self.auc_minority), axis=0)

        self.auc_std = np.std(np.array(self.auc), axis=0)
        self.auc_min_std = np.std(np.array(self.auc_min), axis=0)
        self.auc_max_std = np.std(np.array(self.auc_max), axis=0)
        self.auc_macro_std = np.std(np.array(self.auc_macro), axis=0)
        self.auc_minority_std = np.std(np.array(self.auc_minority), axis=0)

        self.posnegs_avg = np.mean(np.array(self.posnegs), axis=0)
        self.acc_avg = np.mean(np.array(self.auc), axis=0)
        self.steps = np.arange(self.eval_every, len(self.auc_avg)*self.eval_every+self.eval_every, self.eval_every)

    def save_metrics(self, res_dir, dataset, name="ARL"):
        """
        Saves the averaged metrics in a json file.
        """

        metrics = {
            "auc_avg_final": self.auc_avg[-1],
            "auc_std_final": self.auc_std[-1],
            "auc_min_avg_final": self.auc_min_avg[-1],
            "auc_min_std_final": self.auc_min_std[-1],
            "auc_macro_avg_final": self.auc_macro_avg[-1],
            "auc_macro_std_final": self.auc_macro_std[-1],
            "auc_auc_minority_avg": self.auc_minority_avg[-1],
            "auc_auc_minority_std": self.auc_minority_std[-1],
            "tn": self.posnegs_avg[0],
            "fp": self.posnegs_avg[1],
            "fn": self.posnegs_avg[2],
            "tp": self.posnegs_avg[3],
            "posnegs_avg": self.posnegs_avg.tolist(),
            "acc_avg": self.acc_avg.tolist(),
        }
        json.dump(metrics, open("{}{}_{}.json".format(res_dir, dataset, 
            name), 'w'))


class SignificanceMetrics():
    """
    Implements AUC significance tests.
    """

    def calc_pval(self, tp1, tn1, tp2, tn2, auc1, auc2):
        """
        Calculates the P value of two models.
    
        Args:
            tp1: true positives of model 1. 
        tp1: true positives of model 1. 
            tp1: true positives of model 1. 
            tn1: true negatives of model 1.
            tp2: true positives of model 2. 
        tp2: true positives of model 2. 
            tp2: true positives of model 2. 
            tn2: true negatives of model 2.
            auc1: AUC value of model 1.
            auc2: AUC value of model 2.
        """

        # Calculate the standard error.
        se1 = self.standard_error(tp1, tn1, auc1)
        se2 = self.standard_error(tp2, tn2, auc2)

        # Calculate the zscore.
        r = 0
        se_dif = np.sqrt(se1 ** 2 + se2 ** 2 - 2 * r * se1 * se2)
        z = (auc1 - auc2) / se_dif

        # Calculate the pvalue.
        pval = norm.sf(abs(z))*2 
        pval = norm.sf(abs(z))*2 
        pval = norm.sf(abs(z))*2 
        return pval

    def standard_error(self, tp, tn, a):
        """
        Calculates the standard error of an AUC curve.
        First introduced by Hanley and McNeil (1982). And explained here:
        http://www.anaesthetist.com/mnm/stats/roc/Findex.htm

        Args:
            tp: true positives
            tn: true negatives
            a: AUC value
        """
        q1 = a / (2-a)
        q2 = 2 * a**2 / (1 + a)
        se = np.sqrt((a * (1-a) + (tp-1) * (q1 - a ** 2) + \
                     (tn - 1) * (q2 - a ** 2)) / (tp * tn))
        return se






        
        
