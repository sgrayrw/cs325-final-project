import pandas as pd

from util import *


def plot_result(result, metrics, figname):
    for metric, label in metrics.items():
        for version, d in result.items():
            df = pd.DataFrame(d[metric])
            sns.regplot(data=df, x='val', y='accs', ci=None, scatter_kws={'s': np.log(df['counts']) * 4}, label=version)
        plt.xlabel(label)
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
        plt.savefig(f'fig/{figname}_{label}')


def analyze_binary_classification(result):
    metrics = {
        'len': 'Pre-verb sentence length',
        'case_den': 'Case marker density',
    }
    plot_result(result, metrics, 'binary_result')


def analyze_multiple_choice(result):
    metrics = {
        'len': 'Pre-verb sentence length',
        'case_den': 'Case marker density',
        'freq': 'Verb frequency'
    }
    plot_result(result, metrics, 'mc_result')


if __name__ == '__main__':
    sns.set_style('whitegrid')

    binary_result, mc_result = {}, {}
    for result_version in os.listdir('pickle/evaluated_result'):
        folder = os.path.realpath(result_version)
        binary_result[result_version] = load_obj(f'evaluated_result/{result_version}/binary_result')
        mc_result[result_version] = load_obj(f'evaluated_result/{result_version}/mc_result')

    analyze_binary_classification(binary_result)
    analyze_multiple_choice(mc_result)
