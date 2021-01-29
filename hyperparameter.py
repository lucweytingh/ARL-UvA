"""
In hyperparameter.py, the following hyperparameters
are optimized using exhaustive gridsearch:
- batch size: [16, 32, 64, 128, 256]
- learner learning rate: [0.001, 0.01, 0.1, 1]
- adversary learning rate: [0.001, 0.01, 0.1, 1]
"""

import itertools
from sklearn.model_selection import KFold
from collections import defaultdict
import numpy as np
import os
import time

from train import train_for_n_iters
from utils import loadDataset

def main():
    """
    Runs the parameter opimization.
    """

    parameters = {
        "batch_size": [16, 32, 64, 128, 256],
        "learner_lr": [0.001, 0.01, 0.1, 1],
        "adversary_lr": [0.001, 0.01, 0.1, 1],
    }

    for dataset in ["law_school"]:
        start = time.time()
        print("\n" + 40 * "-")
        print(f"tuning hyperparameters for {dataset}")
        print(40 * "-" + "\n")
        best_param, best_auc, step = optimize_parameters(
            parameters, dataset, train_steps=1000, pretrain_steps=250
        )
        stop = time.time()
        print(f"\ntuning took {stop - start:.0f} seconds for {dataset}")
        print(
            f"the best parameters for {dataset} are:\n{best_param}\n"
            f"found on step {step} with an AUC of {best_auc:.3f}\n",
        )

        best_param["train_steps"] = step

        filename = check_filename(f"hyperparameters/{dataset}.txt")

        try:
            with open(filename, "w") as f:
                f.write(str(best_param))
        except:
            print("results file could not be created")


def optimize_parameters(
    parameters, dataset, train_steps=1000, pretrain_steps=250
):
    """
    Returns the best the hyperparameters tuning for the ARL model.

    args:
        parameters: a dictionary with the hyperparameters and their values,
                    e.g. {'batch_size': [32, 64, 256], [...]}.
        dataset: name of the dataset ([toy_data, uci_adult, compas, law]).
    """

    # Load the training data.
    train_dataset = loadDataset(
        dataset=dataset,
        train_or_test="train",
        embedding_size=32,
    )

    # Create the default model parameters.
    model_params = {
        "embedding_size": train_dataset.categorical_embedding_sizes,
        "n_num_cols": len(train_dataset.mean_std.keys()),
        "learner_hidden_units": [64, 32],
        "adversary_hidden_units": [32],
        "batch_size": None,
    }

    lr_params = {"learner": None, "adversary": None}

    cross_val = KFold(n_splits=5)
    steps = None

    # Create a defaultdict for the results.
    params2aucs = defaultdict(list)

    # Get all possible combinations of parameters.
    options = itertools.product(*parameters.values())
    n_options = len(list(itertools.product(*parameters.values())))

    for i, (batch_size, learner_lr, adversary_lr) in enumerate(options, 1):
        iter_start = time.time()
        model_params["batch_size"] = batch_size
        lr_params["learner"] = learner_lr
        lr_params["adversary"] = adversary_lr

        print(
            f"--- ({i}/{n_options}) batch_size: {batch_size}, "
            f"learner_lr: {learner_lr}, adversary_lr: {adversary_lr}"
        )

        # 5-fold cross-validation
        for train_index, test_index in cross_val.split(train_dataset):

            # Get the performance of the model
            metrics = train_for_n_iters(
                train_dataset.get_split(train_index),
                train_dataset.get_split(test_index),
                model_params,
                lr_params,
                average_over=5,
                train_steps=train_steps,
                pretrain_steps=pretrain_steps,
                print_loss=False,
            )
            params2aucs[(batch_size, learner_lr, adversary_lr)].append(
                metrics.auc_avg
            )
            steps = metrics.steps
        iter_stop = time.time()

        mean_best_auc = np.mean(
            params2aucs[(batch_size, learner_lr, adversary_lr)], axis=0
        )

        best_auc_idx = np.argmax(mean_best_auc)

        print(
            f"\t took {iter_stop - iter_start:.0f} seconds | "
            f"best AUC is {mean_best_auc[best_auc_idx]:.3f} on step "
            f"{steps[best_auc_idx]}"
        )

    # Average the folds.
    params2aucs = {
        option: np.mean(aucs, axis=0) for option, aucs in params2aucs.items()
    }

    # Find the highest AUC.
    params = list(params2aucs.keys())
    aucs = np.array(list(params2aucs.values()))

    param_idx, step_idx = np.unravel_index(np.argmax(aucs), aucs.shape)
    best_auc = aucs[param_idx, step_idx]
    best_params = params[param_idx]
    best_step = steps[step_idx]

    # Return the results.
    results = {
        "batch_size": best_params[0],
        "lr_learner": best_params[1],
        "lr_adversary": best_params[2],
    }
    return results, best_auc, best_step


def check_filename(filename):
    """
    Creates directory if the directory to save the file in does not exists.

    Args:
        filename: the filepath.
    """
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
            return filename
        except:
            if filename[0] != "/":
                return check_filename("/" + filename)
            else:
                print(
                    "hyperparameters folder could not be created, saving in main directory"
                )
                return filename.split("/")[1]
    else:
        return filename


if __name__ == "__main__":
    main()
