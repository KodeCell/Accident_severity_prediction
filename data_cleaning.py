import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import encoding
import pickle

raw_data = pd.read_csv('RTA Dataset.csv')
import preprocessing


raw_data.replace('na',np.nan,inplace = True) # since na values are encodes as strings

raw_data['Fitness_of_casuality'].replace('NormalNormal','Normal',inplace = True) # changing duplicate values
raw_data = raw_data[raw_data['Area_accident_occured']!='Rural village areasOffice areas']
raw_data = preprocessing.removeNullValues(raw_data)
raw_data = preprocessing.cleanData(raw_data)
raw_data = preprocessing.changeDateTime(raw_data)
raw_data = encoding.encodeOrdinalValues(raw_data)


le_cols = ['Sex_of_driver','Age_band_of_driver',
           'Vehicle_driver_relation','Type_of_vehicle','Owner_of_vehicle','Service_year_of_vehicle',
           'Area_accident_occured','Lanes_or_Medians','Road_allignment','Types_of_Junction','Road_surface_type',
           'Road_surface_conditions','Light_conditions','Weather_conditions','Type_of_collision','Vehicle_movement',
           'Sex_of_casualty','Work_of_casuality','Fitness_of_casuality','Pedestrian_movement','Cause_of_accident','Day_of_week',
          'Casualty_class',]

#Saving the label encoder in the files label

le = LabelEncoder()
my_path = 'D:/TMLC/project_1/labels/'
for columns in le_cols:
    raw_data[columns] = le.fit_transform(raw_data[columns])
    new_path = f'{my_path}{columns}.pkl'
    output = open(new_path, 'wb')
    pickle.dump(le, output)
    output.close()


df_cleaned = raw_data.drop(columns = ['Defect_of_vehicle','Time','hour'],axis = 1)

df_cleaned.to_csv('D:/TMLC/project_1/Data/df_cleaned.csv',index = False)



