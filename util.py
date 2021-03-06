import os
import pickle
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from Params import Params


def save_obj(obj, name):
    filename = 'pickle/' + name + '.pkl'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    filename = 'pickle/' + name + '.pkl'
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_hist(hist, version):
    hist_df = pd.DataFrame(hist.history)
    hist_json = f'history/{version}.json'
    with open(hist_json, mode='w') as f:
        hist_df.to_json(f)

def load_hist(version) -> pd.DataFrame:
    with open(f'history/{version}.json') as f:
        return pd.read_json(f)


def sort_dict(d):
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}


def freqdict_to_inxdict(freqdict):
    idxdict = {}

    limit = {
        'unigrams': Params.feature_unigram_limit,
        'bigrams': Params.feature_bigram_limit,
        'case_unigrams': float('inf'),
        'case_bigrams': float('inf'),
        'final_case': float('inf')
    }

    for k, d in freqdict.items():
        print('converting', k)
        newd = defaultdict(int)
        idx = 0
        f = f_all = 0
        for _k, _v in d.items():
            f_all += _v
            if idx < limit[k]:
                f += _v
                newd[_k] = idx
            idx += 1
        print(f'f: {f}, f_all: {f_all}')
        idxdict[k] = newd
    return idxdict


# https://github.com/DTrimarchi10/confusion_matrix
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for _ in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize is None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if not xyticks:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)
