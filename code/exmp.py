import os.path
from pathlib import Path
import numpy as np

import qiime2
from qiime2.plugins.diversity.methods import pcoa as pcoa_method, filter_distance_matrix
from qiime2.plugins.longitudinal.visualizers import anova

import seaborn as sns
import scipy.stats
import bisect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skbio.stats.distance import MissingIDError
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

import matplotlib



base_dir = ".."
data_dir = os.path.join(base_dir, 'data', 'exmp1-and-exmp2')
metadata_dir = os.path.join(base_dir, "sample-metadata")

sample_md_fp = os.path.join(metadata_dir, "sample-metadata.tsv")
table_fp = os.path.join(data_dir, "table.qza")
taxonomy_fp = os.path.join(data_dir, "taxonomy-gtdb.qza")
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

def load_taxonomy():
    return qiime2.Artifact.load(taxonomy_fp)

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

def ols_and_anova(dep_variable, project, time_value, base_output_dir, time_column,
                  sample_metadata, uu_dm, wu_dm, faith_pd, shannon, evenness):
    indep_variables = ['faith_pd', 'shannon', 'pielou_e',
                       'Weighted_UniFrac_PC1', 'Weighted_UniFrac_PC2', 'Weighted_UniFrac_PC3',
                       'Unweighted_UniFrac_PC1', 'Unweighted_UniFrac_PC2', 'Unweighted_UniFrac_PC3']
    output_dir = os.path.join(base_output_dir, '%s-%s-%s%s' % (project, dep_variable, time_column, str(time_value)))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    where = "[project]='%s' and [%s]='%s'" % (project, time_column, str(time_value))

    ids_to_keep = sample_metadata.get_ids(where=where)
    sample_metadata = sample_metadata.filter_ids(ids_to_keep=ids_to_keep)

    # make column names compatible for R-like forumulas used in anova
    _df = sample_metadata.to_dataframe()
    _df.index.name = 'sample-id'
    _df = _df.rename(columns={'VO2max-change': 'VO2max_change',
                              'RER-change': 'RER_change',
                              'row-change': 'row_change',
                              'bench-press-change': 'bench_press_change',
                              '3RM-squat-change': 'three_rep_max_squat_change'})

    # drop columns that don't have necessary data
    if project == 'exmp1':
        _df = _df[['VO2max_change', 'RER_change']].dropna().astype(np.float)
    elif project == 'exmp2':
        _df = _df[['row_change', 'bench_press_change', 'three_rep_max_squat_change']].dropna().astype(np.float)
    else:
        raise ValueError("Project must be exmp1 or exmp2, but %s was provided." % project)
    sample_metadata = qiime2.Metadata(_df)

    uu_dm = filter_distance_matrix(uu_dm, metadata=sample_metadata).filtered_distance_matrix
    wu_dm = filter_distance_matrix(wu_dm, metadata=sample_metadata).filtered_distance_matrix

    wu_pcoa = pcoa_method(wu_dm).pcoa
    wu_pcoa = wu_pcoa.view(qiime2.Metadata).to_dataframe()[['Axis 1', 'Axis 2', 'Axis 3']]
    wu_pcoa = wu_pcoa.rename(columns={'Axis 1': 'Weighted_UniFrac_PC1',
                                      'Axis 2': 'Weighted_UniFrac_PC2',
                                      'Axis 3': 'Weighted_UniFrac_PC3'})
    sample_metadata = sample_metadata.merge(qiime2.Metadata(wu_pcoa))

    uu_pcoa = pcoa_method(uu_dm).pcoa
    uu_pcoa = uu_pcoa.view(qiime2.Metadata).to_dataframe()[['Axis 1', 'Axis 2', 'Axis 3']]
    uu_pcoa = uu_pcoa.rename(columns={'Axis 1': 'Unweighted_UniFrac_PC1',
                                      'Axis 2': 'Unweighted_UniFrac_PC2',
                                      'Axis 3': 'Unweighted_UniFrac_PC3'})
    sample_metadata = sample_metadata.merge(qiime2.Metadata(uu_pcoa))

    sample_metadata = sample_metadata.merge(faith_pd.view(qiime2.Metadata))
    sample_metadata = sample_metadata.merge(shannon.view(qiime2.Metadata))
    sample_metadata = sample_metadata.merge(evenness.view(qiime2.Metadata))


    df = sample_metadata.to_dataframe()
    df = sm.add_constant(df)

    dep_variable_histogram = df[dep_variable].hist().figure
    dep_variable_histogram.savefig(os.path.join(output_dir, 'histogram.pdf'))

    mod = sm.OLS(df[dep_variable], df[['const'] + indep_variables])
    res = mod.fit()
    ols_result_summary = res.summary()
    with open(os.path.join(output_dir, 'ols.csv'), 'w') as fh:
        fh.write(ols_result_summary.as_csv())

    formula = "%s ~ %s" % (dep_variable, ' + '.join(indep_variables))
    anova_visualization = anova(metadata=qiime2.Metadata(df), formula=formula).visualization
    anova_visualization.save(os.path.join(output_dir, 'anova.qzv'))

    return (dep_variable_histogram, ols_result_summary, anova_visualization,
            sample_metadata)
