from sklearn.preprocessing import LabelEncoder
import pandas as pd

def encodeCategories(df,cols):
    le = LabelEncoder()
    for column in cols:
        df[column] = le.fit_transform(df[column])

    return df
def encodeOrdinalValues(df):
    Education = {
        'Illiterate': 0,
        'Writing & reading': 1,
        'Elementary school': 2,
        'Junior high school': 3,
        'High school': 4,
        'Above high school': 5,
        'Other': 6
    }
    Driving_exp = {
        'Other': 0,
        'No Licence': 1,
        'Below 1yr': 2,
        '1-2yr': 3,
        '2-5yr': 4,
        '5-10yr': 5,
        'Above 10yr': 6
    }

    casuality_age = {
        '5': 0,
        'Under 18': 1,
        '18-30': 2,
        '31-50': 3,
        'Over 51': 4
    }
    df['Education'] = df['Educational_level'].map(Education).astype(int)
    df['Driving_exp'] = df['Driving_experience'].map(Driving_exp).astype(int)
    df['Casualty_age'] = df['Age_band_of_casualty'].map(casuality_age).astype(int)

    return df
