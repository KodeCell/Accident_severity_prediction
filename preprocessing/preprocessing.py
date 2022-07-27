import pandas as pd
import numpy as np


def removeNullValues(df):

    for cols in df.columns:
        if df[cols].isnull().sum() > 1:
            df[cols] = df[cols].fillna(df[cols].mode()[0])
    return df
def cleanData(df):
    #function to change different termanology into one

    df.replace('other', 'Other', inplace=True)
    df.replace('unknown', 'Unknown', inplace=True)
    df.replace('Unknown', 'Other', inplace=True)
    df['Defect_of_vehicle'].replace('No defect', 0, inplace=True)

    return df

def changeDateTime(df):
    df['Time'] = pd.to_datetime(df['Time'])

    df['hour'] = df['Time'].dt.hour

    df['Sin_hour'] = np.sin(2 * np.pi * df['hour'] / max(df['hour']))
    df['Cos_hour'] = np.cos(2 * np.pi * df['hour'] / max(df['hour']))

    return df


