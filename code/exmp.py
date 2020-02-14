import os.path
import numpy as np

import qiime2

import seaborn as sns
import scipy.stats
import bisect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skbio.stats.distance import MissingIDError
from statsmodels.stats.multitest import multipletests

import matplotlib

base_dir = ".."
data_dir = os.path.join(base_dir, 'data', 'exmp1-and-exmp2')
metadata_dir = os.path.join(base_dir, "sample-metadata")

sample_md_fp = os.path.join(metadata_dir, "sample-metadata.tsv")
table_fp = os.path.join(data_dir, "table.qza")
phylogeny_fp = os.path.join(data_dir, "rooted-tree.qza")
cm_path = os.path.join(data_dir, 'cm')


# the following files can be created by group-weeks-to-period.ipynb
sample_md_grouped_by_period_fp = \
    os.path.join(metadata_dir, "sample-metadata-grouped-by-period.tsv")
table_grouped_by_period_fp = \
    os.path.join(data_dir, "table-grouped-by-period.qza")

cm_grouped_by_period_path = os.path.join(data_dir, 'cm-grouped-by-period')

def load_table():
    return qiime2.Artifact.load(table_fp)

def load_table_grouped_by_period():
    return qiime2.Artifact.load(table_grouped_by_period_fp)

def load_sample_metadata():
    return qiime2.Metadata.load(sample_md_fp)

def load_sample_metadata_grouping():
    # resulting ids are original sample ids
    sample_metadata = load_sample_metadata().to_dataframe()
    sample_metadata['period'] = list(map(week_to_period,
                                         sample_metadata['week']))
    grouped_sample_metadata = sample_metadata[
        ['period', 'subject-id', 'project', 'activity', 'exclude',
         'VO2max-change', 'RER-change', 'row-change', 'bench-press-change',
         '3RM-squat-change']]
    grouped_sample_metadata['subject-id-period'] = \
         ['-'.join(map(str,e)) \
              for e in zip(grouped_sample_metadata['subject-id'],
                           grouped_sample_metadata['period'])]
    return qiime2.Metadata(grouped_sample_metadata)

def load_sample_metadata_grouped_by_period():
    # resulting ids are subject-period (corresponding to grouped table ids)
    grouped_sample_metadata = load_sample_metadata_grouping().to_dataframe()
    grouped_sample_metadata = \
        grouped_sample_metadata.drop_duplicates(subset='subject-id-period')
    grouped_sample_metadata = \
        grouped_sample_metadata.set_index('subject-id-period')
    grouped_sample_metadata.index.name = 'sample-id'
    return qiime2.Metadata(grouped_sample_metadata)

def load_phylogeny():
    return qiime2.Artifact.load(phylogeny_fp)

def week_to_period(week):
    week = float(week)
    if np.isnan(week):
        return week
    elif week < 4:
        return '1'
    elif week < 7:
        return '2'
    elif week < 9:
        return '3'
    elif week < 12:
        return '4'
    else:
        return '5'




alphas = [(0.001, '***'), (0.01, '**'), (0.05, '*')]

font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 8}
matplotlib.rc('font', **font)

sns.set_style("whitegrid")

def get_sig_text(p, alphas, null_text=""):
    if np.isnan(p):
        return null_text
    alphas.sort()
    if p >= alphas[-1][0]:
        return 'ns'
    sorted_location = bisect.bisect([e[0] for e in alphas], p)
    return alphas[sorted_location][1]

def plot_week_data(df, metric, time_column, label_axes=True, output_figure_filepath=None):
    df[time_column] = pd.to_numeric(df[time_column], errors='coerce')
    df[metric] = pd.to_numeric(df[metric], errors='coerce')

    df = df.sort_values(by=time_column)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax = sns.boxplot(data=df, x=time_column, y=metric, color='white', ax=ax)
    ax = sns.swarmplot(data=df, x=time_column, y=metric, color='black', ax=ax)

    x0 = np.min(df[time_column]) - 1
    x1 = np.max(df[time_column]) + 1

    if not label_axes:
        ax.set_xlabel('')
        ax.set_ylabel('')

    if output_figure_filepath is not None:
        fig.savefig(output_figure_filepath, dpi = (300))
    else:
        return fig

# Change in distance to donor from time zero is different than zero
# (positive t means more different than donor, negative t means more similar to donor)
def tabulate_week_to_reference_week_paired_stats(df, metric, reference_time, time_column, test_fn=scipy.stats.mannwhitneyu):
    results = []
    df[time_column] = pd.to_numeric(df[time_column], errors='coerce')
    df = df.sort_values(by=time_column)
    weeks = df[time_column].unique()
    for i in weeks:
        reference_time_metric = df[df[time_column] == reference_time][metric]
        time_i_metric = df[df[time_column] == i][metric]
        t, p = test_fn(reference_time_metric, time_i_metric)
        results.append((i, len(time_i_metric), np.median(time_i_metric), t, p))
    result = pd.DataFrame(results,
                          columns=[time_column, 'n', metric, 'test-statistic', 'p-value']
                         )
    result = result.set_index(time_column)
    result['q-value'] = multipletests(result['p-value'])[1]
    return result

def plot_week_data_with_stats(sample_md, metric, time_column, hue=None, alphas=alphas, reference_time=1,
                              output_figure_filepath=None, output_table_filepath=None):
    fig = plot_week_data(sample_md, metric, time_column, label_axes=True)
    stats = tabulate_week_to_reference_week_paired_stats(sample_md, metric, reference_time, time_column)
    ymax = fig.axes[0].get_ylim()[1]
    stats.sort_index()
    for i, w in enumerate(stats.index):
        t, q = stats['test-statistic'][w], stats['q-value'][w]
        sig_text = get_sig_text(q, alphas)
        fig.axes[0].text(i, 1.02*ymax, sig_text, ha='center', va='center')
    if output_table_filepath is not None:
        stats.to_csv(output_table_filepath)
    if output_figure_filepath is not None:
        fig.savefig(output_figure_filepath, dpi = (300))
    else:
        return fig
