import os.path
import numpy as np

import qiime2

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
