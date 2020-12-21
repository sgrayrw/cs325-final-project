import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import *


def analyze_binary_classification(ver):
    binary_result = load_obj(f'binary_result{ver}')
    len_df = pd.DataFrame(binary_result['len'])
    case_den_df = pd.DataFrame(binary_result['case_den'])

    sns.regplot(data=len_df, x='lens', y='accs', scatter_kws={'s': np.log(len_df['counts']) * 4})
    plt.xlabel('Pre-verb sentence length')
    plt.ylabel('Accuracy')
    plt.show()

    sns.regplot(data=case_den_df, x='dens', y='accs', scatter_kws={'s': np.log(case_den_df['counts']) * 4})
    plt.xlabel('Case marker density')
    plt.ylabel('Accuracy')
    plt.show()

    conf = binary_result['conf']
    make_confusion_matrix(conf, group_names=['True Neg', 'False Pos', 'False Neg', 'True Pos'])
    plt.show()


def analyze_multiple_choice(ver):
    mc_result = load_obj(f'mc_result{ver}')
    len_df = pd.DataFrame(mc_result['len'])
    case_den_df = pd.DataFrame(mc_result['case_den'])
    freq_df = pd.DataFrame(mc_result['freq'])

    sns.regplot(data=len_df, x='lens', y='accs', scatter_kws={'s': np.log(len_df['counts']) * 4})
    plt.xlabel('Pre-verb sentence length')
    plt.ylabel('Accuracy')
    plt.show()

    sns.regplot(data=case_den_df, x='dens', y='accs', scatter_kws={'s': np.log(case_den_df['counts']) * 4})
    plt.xlabel('Case marker density')
    plt.ylabel('Accuracy')
    plt.show()

    sns.regplot(data=freq_df, x='freqs', y='accs', scatter_kws={'s': np.log(freq_df['counts']) * 4})
    plt.xlabel('Verb frequency')
    plt.ylabel('Accuracy')
    plt.show()


if __name__ == '__main__':
    ver = ''
    sns.set_style('whitegrid')
    analyze_binary_classification(ver)
    analyze_multiple_choice(ver)
