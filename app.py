
import os
import sys
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PACKAGE_ROOT not in sys.path:
    sys.path.append(PACKAGE_ROOT)

from src.pipeline.predict_pipeline import customData , predictpipeline

application= Flask(__name__)

app= application

## Route for  a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method== 'GET':
      return render_template('home.html')
    else:
       data = customData(
          gender = request.form.get('gender'),
          race_ethnicity = request.form.get('ethnicity'),
          parental_level_of_education = request.form.get('parental_level_of_education'),
          lunch = request.form.get('lunch'),
          test_preparation_course = request.form.get('test_preparation_course'),
          reading_score=request.form.get('reading_score'),
          writing_score=request.form.get('writing_score')
       )

       pred_df=data.get_data_as_a_dataframe()
       print(pred_df)

       predict_pipeline= predictpipeline()
       results= predict_pipeline.predict(pred_df)
       return render_template('home.html',results= results[0])

if __name__ =="__main__" :
   app.run(host='0.0.0.0', debug= True)