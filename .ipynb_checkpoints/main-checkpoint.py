from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
import numpy as np
import pickle

app = Flask(__name__)

loaded_svm_model = joblib.load('final_svm_model.pkl')
required_columns = ['billing_type', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 'fraud_consumption',
                   'SERVICE_STATUS', 'POWER_SUSCRIBED', 'TARIFF', 'ACTIVITY_CMS', 'READWITH', 'SEGMENT',
                   'agency', 'zone']
model = pickle.load(open('final_svm_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/single-customer')
def single_customer():
    return render_template('index.html')

@app.route('/predict_one_customer',methods=['POST'])
def predict_one_customer():
    '''
    For rendering results on HTML GUI
    '''
        #int_features = [int(x) for x in request.form.values()]
    int_features = [ x for x in request.form.values()]

    contract_no = int(int_features[0])
    new_consumption = float(int_features[1])

    # contract_no = 200050059
    #new_consumption = 7000.0
    # print(contract_no)
    print(new_consumption)
    
    
    #final_features = [np.array(int_features)]
    rawData = pd.read_csv('balanced_data.csv')
    #print(rawData)

    rawData=rawData.loc[(rawData['contract']==contract_no)]
    print(rawData)

    #rawData['fraud_flag'] = np.where(rawData['fraud_flag'] == 1.0, 1, 0)
    rawData['fraud_consumption'] = new_consumption
    # rawData['1/1/2019'] = new_consumption
    print("rawData : ",rawData)

    rawData.replace('', np.nan, inplace=True)
    df1 = rawData[required_columns]
    imputer = SimpleImputer(strategy='mean')
    # df_imputed = pd.DataFrame(imputer.fit_transform(df1), columns=df1.columns)
    prediction = loaded_svm_model.predict(df1)
    print("prediction :", prediction)
    rawData['prediction'] = prediction
    
   

    output = round(prediction[0], 2)
    if(output==0):
        res="Normal Consumption"
    else:
        res="Fraud Consumption"

    return render_template('index.html', prediction_text='PREDICTION IS {}'.format(res))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

@app.route('/multiple-customers')
def multiple_customers():
    return render_template('csv_input.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('csv_input.html', error="No file found")
    
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return render_template('csv_input.html', error="Only CSV files are supported")
    df = pd.read_csv(file)
    df.replace('', np.nan, inplace=True)
    df1 = df[required_columns]
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df1), columns=df1.columns)
    predictions = loaded_svm_model.predict(df_imputed)
    df['prediction'] = predictions
    output_filename = 'predictions.csv'
    df.to_csv(output_filename, index=False)
    return send_file(output_filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
