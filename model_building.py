import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle



data = pd.read_csv('Data/df_cleaned.csv')
y = data['Accident_severity']
x = data.drop(columns = ['Accident_severity','Pedestrian_movement','Fitness_of_casuality','Work_of_casuality','Sex_of_casualty','Casualty_class','Age_band_of_casualty',
                         'Cause_of_accident','Age_band_of_driver','Sex_of_driver','Sin_hour','Cos_hour','Educational_level','Driving_experience','Age_band_of_casualty','Casualty_age'],axis = 1)
scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

pickle.dump(scaler, open('scaler.pkl', 'wb'))


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)


#since we are building a model on a imbalanced classes we doing upsampling of minority class
#train set and check how it goes on test set
def upsampling(x_train,y_train):

    oversample = SMOTE()
    x_train, y_train = oversample.fit_resample(x_train, y_train)
    return x_train,y_train
x_train,y_train = upsampling(x_train,y_train)

rf_tuned = RandomForestClassifier(ccp_alpha= 0.0,
  max_depth= 9,
  min_samples_split= 4,
  n_estimators= 200)

rf_tuned.fit(x_train,y_train)


import pickle
pickle.dump(rf_tuned,open('final_model.pickle','wb'))
