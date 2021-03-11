import pandas as pd

from util import *

metrics = {
    'len': 'Pre-verb sentence length',
    'case_den': 'Case marker density',
    'freq': 'Verb frequency',
    'percent_revealed': 'Percentage revealed'
}

def plot_result(result, figname):
    for metric, label in metrics.items():
        plot = False
        for version, d in result.items():
            if metric in d:
                plot = True
                df = pd.DataFrame(d[metric])

                # # limit data range
                if metric == 'len':
                    df = df[df['val'] < 60]

                sns.regplot(data=df, x='val', y='accs', ci=None, scatter_kws={'s': np.log(df['counts']) * 4}, label=version)
        if plot:
            plt.title(f'{figname}: {label}')
            plt.xlabel(label)
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()
            plt.savefig(f'fig/{figname}_{label}')


def analyze_binary_classification(result):
    plot_result(result, 'binary_result')


def analyze_multiple_choice(result):
    plot_result(result, 'mc_result')


def analyze_history(version):
    df = load_hist(version)
    df['epoch'] = df.index

    fig, axes = plt.subplots(nrows=2, ncols=2)
    df.plot(x='epoch', y='loss', ax=axes[0, 0])
    df.plot(x='epoch', y='accuracy', ax=axes[1, 0])
    df.plot(x='epoch', y='val_loss', ax=axes[0, 1])
    df.plot(x='epoch', y='val_accuracy', ax=axes[1, 1])
    plt.show()


if __name__ == '__main__':
    sns.set_style('whitegrid')

    # binary_result, mc_result = {}, {}
    # for result_version in os.listdir('pickle/evaluated_result'):
    #     folder = os.path.realpath(result_version)
    #     binary_result[result_version] = load_obj(f'evaluated_result/{result_version}/binary_result')
    #     mc_result[result_version] = load_obj(f'evaluated_result/{result_version}/mc_result')
    #
    # analyze_binary_classification(binary_result)
    # analyze_multiple_choice(mc_result)

    binary_result = {'lstm': load_obj('evaluated_result/lstm/binary_percent_result')}
    analyze_binary_classification(binary_result)
    mc_result = {'lstm': load_obj('evaluated_result/lstm/mc_percent_result')}
    analyze_multiple_choice(mc_result)

    analyze_history('lstm')
