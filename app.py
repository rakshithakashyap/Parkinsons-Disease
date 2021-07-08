from flask import Flask, request , render_template
from werkzeug.utils import secure_filename
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


app = Flask(__name__ , static_url_path='/static')
# model=pickle.load(open('parkinson.data','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/output', methods = ['GET', 'POST'])
def output():

    if request.method == 'POST':
        f = request.files['ipfile']
        f.save(secure_filename(f.filename))
        file=open(f.filename,'r')
        RawData=file.read().strip()
        Data=RawData.split("=")
        str=""
        for d in Data[1::]:
            str=str+re.sub('[\t]*\n',' ',d).split(' ')[0]+","
        content=str.rstrip(str[-1])
        result=parkinson(content)
        return render_template('output.html', text=result)
  
def parkinson(inputStr):
    df=pd.read_csv('parkinsons.data')
    features=df.loc[:,df.columns!='status'].values[:,1:]
    labels=df.loc[:,'status'].values
    print(labels[labels==1].shape[0], labels[labels==0].shape[0])
    scaler=MinMaxScaler((-1,1))
    x=scaler.fit_transform(features)
    y=labels
    x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)
    model=XGBClassifier()
    model.fit(x_train,y_train)
    #y_pred=model.predict(x_test)
    inputData=inputStr.split(",")
    finalData=tuple(map(float,inputData))
    ''''inputArr=inputStr.split(',')
    inputData=tuple(float(inputArr))'''
    inpNumpy = np.asarray(finalData)
    inpReshapped = inpNumpy.reshape(1, -1)
    stdData = scaler.transform(inpReshapped)
    prediction = model.predict(stdData)

    if (prediction[0] == 0):
        result="The person is healthy"
    else:
        result="The person is suffering from Parkinson's disease"
    return result


if __name__=="__main__":
    app.run(debug=True) 