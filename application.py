import flask
import os
import sys
import cPickle as pickle
import sklearn
import pandas as pd
from flask import Flask, request, render_template
from sklearn.ensemble import GradientBoostingClassifier

application = flask.Flask(__name__)

with open('data/my_model.pkl') as f:
    model = pickle.load(f)

def gender_columns(gender_sel):
    if gender_sel == 'Male':
        male = 1.0
        female = 0.0
    else:
        female = 1.0
        male = 0.0
    return female, male

def location_code(location):
    if location == 'Suburban':
        suburban = 1.0
        rural = 0.0
        urban = 0.0
    elif location == 'Rural':
        suburban = 0.0
        rural = 1.0
        urban = 0.0
    else:
        suburban = 0.0
        rural = 0.0
        urban = 1.0
    return rural, suburban, urban

def vehicle_size(veh_size):
    if veh_size == 'Large':
        large = 1.0
        medsize = 0.0
        small = 0.0
    elif veh_size == 'Small':
        large = 0.0
        medsize = 0.0
        small = 1.0
    else:
        large = 0.0
        medsize = 1.0
        small = 0.0
    return large, medsize, small

# home page
@application.route('/')
def index():
    return flask.render_template('index.html', flask_debug=application.debug)

@application.route('/readme', methods=['GET'])
def readme():
    return flask.render_template('readme.html', flask_debug=application.debug)

@application.route('/predict', methods=['POST'])
def predict():
    income = str(request.form['income'])
    premium = str(request.form['premium'])
    month = str(request.form['month'])
    gender = str(u", ".join(request.form.getlist('gender')))
    location = str(u", ".join(request.form.getlist('location')))
    vehicle  = str(u", ".join(request.form.getlist('vehicle')))

    data = pd.DataFrame([gender_columns(gender)[0], gender_columns(gender)[1],
        location_code(location)[0],location_code(location)[1],location_code(location)[2],
        vehicle_size(vehicle)[0],vehicle_size(vehicle)[1],vehicle_size(vehicle)[2],
        income, premium, month]).T

    pred = "{0:.2f}%".format(round(model.predict_proba(data)[0][1]*100,4))
    return flask.render_template('predict.html').format(pred)

if __name__ == '__main__':
    #application.run(host='0.0.0.0', threaded=True, debug=True)
    application.run()
