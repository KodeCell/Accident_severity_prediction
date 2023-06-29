# Accident_severity_prediction
A Deployed ML model which will tell you the severity of the accident.

## ❓Problem Statement
This data set is collected from Addis Ababa Sub-city police departments for master's research work. The data set has been prepared from manual records of road traffic accidents of the year 2017-20. All the sensitive information has been excluded during data encoding and finally it has 32 features and 12316 instances of the accident. Then it is preprocessed and for identification of major causes of the accident by analyzing it using different machine learning classification algorithms.<br/>
The target feature is Accident_severity which is a multi-class variable. The task is to classify this variable based on the other 31 features

## 📙Dataset

* The dataset for this project can be found here : https://github.com/KodeCell/Accident_severity_prediction/blob/master/RTA%20Dataset.csv
* The dataset contains 31 columns explaining the accident occured on a day.<br />
we re supposed to find the severity of the accident.

## 📷Preview
Below is the preview of the app 


https://user-images.githubusercontent.com/66371234/181154918-d342ad96-13c6-456f-bff1-68bc2034cd2d.mp4

## 🧹Data Cleaning
* Data cleaning is the very first step of after getting the data from the source.<br/>
* The data contained many null values which were then taken care of and there were some instances where the null values are already filled up with different values meaning the same.( Unkown,nan )<br/>
* Almost all the columns except 3-4 columns were categorical so we need to apply encoding for the categories.
* Ecnoder classes were then saved in *labels* folder to access them during web deployment
* Cleaned data was then saved into the *df_cleaned* under *data* folder.

## 💻Model Building
* The baseline model building approach is shared in the ipynb file where many model were trained and were tested.
* Random Forest Classifier was chosen and the final model is trained using tuned hyperparameters of Random Forest.

## 👩‍💻App
* Flask was used in this project to create an api and web application
* The saved model was then loaded into the app which was used to predict the outupt for the the information passed by the user on the web page.
* Inputs were also encoded using the saved classes of the encoder.
