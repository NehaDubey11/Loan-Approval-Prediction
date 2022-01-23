# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 00:00:11 2022

@author: 91639
"""

from flask import Flask,escape , request, render_template
import pickle

app = Flask(__name__)
model=pickle.load(open('lr_pickle','rb'))
@app.route("/")
def hello_world():
    return render_template('index.html')
@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        married=request.form['married']
        credit=request.form['credit']
        ApplicantIncome=request.form['ApplicantIncome']
        LoanAmount=request.form['LoanAmount']
        
        ApplicantIncome = float(request.form['ApplicantIncome'])
    
        LoanAmount = float(request.form['LoanAmount'])
        if(married=="Yas"):
            married_yes = 1
        else:
            married_yes=0
            
            
        prediction = model.predict([[credit, ApplicantIncome,LoanAmount, married_yes]])

        # print(prediction)

        if(prediction=="N"):
            prediction="No"
        else:
            prediction="Yas"


        return render_template("prediction.html", prediction_text="loan status is {}".format(prediction))
        
    else:
        return render_template('prediction.html')

if __name__=="__main__":
    app.run(debug=True)