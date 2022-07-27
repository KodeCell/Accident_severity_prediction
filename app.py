import pandas as pd
from flask import Flask, render_template, jsonify, request
import numpy as np
import pickle
model = pickle.load(open('final_model.pickle','rb'))


app = Flask(__name__)
@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods = ['POST','GET'])
def predict():
    if request.method == 'POST':
        Day_of_week = request.form['Day_of_week']
        Educational_level = int(request.form['Educational_level'])
        Vehicle_driver_relation = request.form['Vehicle_driver_relation']
        Driving_experience = int(request.form['Driving_experience'])
        Type_of_vehicle = request.form['Type_of_vehicle']
        Owner_of_vehicle = request.form['Owner_of_vehicle']
        Service_year_of_vehicle = request.form['Service_year_of_vehicle']
        Area_accident_occured = request.form['Area_accident_occured']
        Lanes_or_medians = request.form['Lanes_or_Medians']
        Road_allignment = request.form['Road_allignment']
        Types_of_junction = request.form['Types_of_Junction']
        Road_surface_type = request.form['Road_surface_type']
        Road_surface_conditions = request.form['Road_surface_conditions']
        Light_conditions = request.form['Light_conditions']
        Weather_conditions = request.form['Weather_conditions']
        Type_of_collision  = request.form['Type_of_collision']
        Number_of_vehicles_involved = int(request.form['Number_of_vehicles_involved'])
        Number_of_casualties = int(request.form['Number_of_casualties'])
        Vehicle_movement = request.form['Vehicle_movement']
        Casualty_severity =int(request.form['Casualty_severity'])

        encoding_cols = [Day_of_week,Vehicle_driver_relation,Type_of_vehicle,
       Owner_of_vehicle,Service_year_of_vehicle,Area_accident_occured,
       Lanes_or_medians,Road_allignment,Types_of_junction,
       Road_surface_type,Road_surface_conditions,Light_conditions,
       Weather_conditions,Type_of_collision,
       Number_of_vehicles_involved,Number_of_casualties,
       Vehicle_movement,Casualty_severity,Educational_level,Driving_experience,]


        cols = ['Day_of_week', 'Vehicle_driver_relation', 'Type_of_vehicle',
       'Owner_of_vehicle', 'Service_year_of_vehicle', 'Area_accident_occured',
       'Lanes_or_Medians', 'Road_allignment', 'Types_of_Junction',
       'Road_surface_type', 'Road_surface_conditions', 'Light_conditions',
       'Weather_conditions', 'Type_of_collision',
       'Number_of_vehicles_involved', 'Number_of_casualties',
       'Vehicle_movement', 'Casualty_severity', 'Education', 'Driving_exp',]

        import pickle
        final_col = []
        for (i, j) in zip(cols, encoding_cols):
            try:
                path = 'D:/TMLC/project_1/labels/'
                final_path = f'{path}{i}.pkl'
                print(final_path)
                pkl_file = open(final_path, 'rb')
                le = pickle.load(pkl_file)
                pkl_file.close()
                js = int(le.transform([j]))
                final_col.append(js)
            except:
                final_col.append(j)

        sc = pickle.load(open('scaler.pkl', 'rb'))
        input = sc.transform(np.array(final_col).reshape(1,-1))
        model = pickle.load(open('final_model.pickle', 'rb'))
        prediction = model.predict(np.array(input).reshape(1,-1))


    return render_template('result.html',prediction = prediction[0])


if __name__ == '__main__':
    app.run(debug=True)



