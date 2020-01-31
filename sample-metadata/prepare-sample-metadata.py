#!/usr/bin/env python
# This code was adapted from:
# https://developers.google.com/sheets/api/quickstart/python

import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

import pandas as pd
import numpy as np
from qiime2 import Metadata
from qiime2.plugins.metadata.actions import tabulate

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

def get_sheet_as_df(spreadsheet_id, sheet_name, index):
    """Shows basic usage of the Sheets API.
    Prints values from a sample spreadsheet.
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds)

    # Call the Sheets API
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=spreadsheet_id,
                                range=sheet_name).execute()
    values = result.get('values', [])
    df = pd.DataFrame(values[1:], columns=values[0]).set_index(index)
    return df

if __name__ == '__main__':
    spreadsheet_id = '1297wYBudjXQmoxJE6UzVhK0pYF7gGJhF39gUYF1Wxj8'

    # join exmp1 subject data, which have different columns
    exmp1_subject_data = get_sheet_as_df(
                            spreadsheet_id,
                            'exmp1-subject-data', index='subject-id')
    exmp1_subject_data = exmp1_subject_data.join(
                    get_sheet_as_df(spreadsheet_id, 'exmp1-weekly-steps', index='subject-id'),
                    on='subject-id')
    exmp1_subject_data = exmp1_subject_data.join(
                    get_sheet_as_df(spreadsheet_id, 'exmp1-weekly-nmvpa', index='subject-id'),
                    on='subject-id')

    # extend with exmp2 subject data, which has no overlapping subject-ids and
    # some of the same columns
    exmp2_subject_data = get_sheet_as_df(spreadsheet_id, 'exmp2-subject-data', index='subject-id')
    subject_data = pd.concat([exmp1_subject_data, exmp2_subject_data],
                             sort=False)


    sample_metadata = get_sheet_as_df(spreadsheet_id, 'combined-minimal', index='sample-id')
    subject_data_indexed_by_sample_id = subject_data.loc[sample_metadata['subject-id']].set_index(sample_metadata.index)
    sample_metadata = sample_metadata.join(subject_data_indexed_by_sample_id, on='sample-id', lsuffix='', rsuffix='_drop_me')
    sample_metadata = sample_metadata.replace(r'^\s*$', np.nan, regex=True)

    sample_metadata = Metadata(sample_metadata)
    sample_metadata.save('sample-metadata.tsv')
    tabulate(sample_metadata).visualization.save('sample-metadata.qzv')
