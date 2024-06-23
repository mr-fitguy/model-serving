from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
import numpy as np
import pickle

app = Flask(__name__)
#test





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
    '''
    Fraud Consumption Inputs
    # contract_no = 200236826
    #new_consumption = 7000.0

    Normal Consumption Inputs
    # contract_no = 200236826
    #new_consumption = 10
    

    Fraud Consumption Inputs
    # contract_no = 200236832
    #new_consumption = 7000.0

    Normal Consumption Inputs
    # contract_no = 200236832
    #new_consumption = 10.0

    '''
    # contract_no = 200236826
    #new_consumption = 7000.0
    print(contract_no)
    print(new_consumption)
    
    
    #final_features = [np.array(int_features)]
    rawData = pd.read_csv('balanced_data.csv')


    rawData=rawData.loc[(rawData['contract']==contract_no)]
    #print(rawData)

    #rawData['fraud_flag'] = np.where(rawData['fraud_flag'] == 1.0, 1, 0)
    rawData['fraud_consumption'] = new_consumption
    # rawData['1/1/2019'] = new_consumption
    print("rawData : ",rawData)
    required_columns_model = ['contract','invoice_type','billing_type', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 'fraud_consumption',
                   'SERVICE_STATUS', 'POWER_SUSCRIBED', 'TARIFF', 'ACTIVITY_CMS', 'READWITH', 'SEGMENT',
                   'agency', 'zone','block']
    rawData.replace('', np.nan, inplace=True)
    df1 = rawData[required_columns_model]
    df1.set_index('contract', inplace = True)

    #imputer = SimpleImputer(strategy='mean')
    #df_imputed = pd.DataFrame(imputer.fit_transform(df1), columns=df1.columns)
    # load model from file
    loaded_model = pickle.load(open("pima.pickle.new.dat", "rb"))
    y_pred1 = loaded_model.predict(df1)
    prediction = [round(value) for value in y_pred1]
    print("prediction :", prediction)
    df1['prediction'] = prediction
    


    output = round(prediction[0], 0)
    
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

    model = pickle.load(open('final_svm_model.pkl', 'rb'))
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
    df.sort_values("contract", inplace=True)

    # dropping ALL duplicate values
    df.drop_duplicates(subset="contract",
                     keep=False, inplace=True)


    rawData = pd.read_csv('balanced_data.csv')
    rawData.sort_values("contract", inplace=True)

    # dropping ALL duplicate values
    rawData.drop_duplicates(subset="contract",
                     keep=False, inplace=True)


    #rawData.set_index('contract', inplace = True)
    

    

    df.replace('', np.nan, inplace=True)
    #df.set_index('contract', inplace = True)

    df_merged = df.merge(rawData, on='contract', how='right')
    df_merged = df_merged.rename(columns = {'fraud_consumption_x':'fraud_consumption'})


    
    required_columns_model = ['contract','invoice_type','billing_type', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 'fraud_consumption',
                   'SERVICE_STATUS', 'POWER_SUSCRIBED', 'TARIFF', 'ACTIVITY_CMS', 'READWITH', 'SEGMENT',
                   'agency', 'zone','block']
    df_merged.replace('', np.nan, inplace=True)
    df_merged = df_merged[required_columns_model]
    df_merged.set_index('contract', inplace = True)


    
    loaded_model = pickle.load(open("pima.pickle.new.dat", "rb"))
    y_pred1 = loaded_model.predict(df_merged)
    prediction = [round(value) for value in y_pred1]
    #print("prediction :", prediction)
    df_merged['prediction'] = prediction

    

    output_filename = 'predictions.csv'
    df_merged.to_csv(output_filename, index=True)
    return send_file(output_filename, as_attachment=True)

# @app.route('/bulk-predict', methods=['GET', 'POST'])
# def bulk_predict():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return 'No file part'
#         file = request.files['file']
#         if file.filename == '':
#             return 'No selected file'
#         if file:
#             user_df = pd.read_csv(file)
#             if 'contract' not in user_df.columns or 'fraud_consumption' not in user_df.columns:
#                 return 'Invalid file format. The CSV must contain "contract_no" and "fraud_consumption" columns.'

#             main_df = pd.read_csv('data_final_v3.csv')

#             merged_df = pd.merge(main_df, user_df, how='inner', left_on='contract', right_on='contract')
#             merged_df['fraud_consumption'] = merged_df['fraud_consumption_y']

#             df1 = merged_df[required_columns]
#             imputer = SimpleImputer(strategy='mean')
#             df_imputed = pd.DataFrame(imputer.fit_transform(df1), columns=df1.columns)

#             predictions = loaded_svm_model.predict(df_imputed)
#             user_df['prediction']=predictions
#             print(len(predictions))
#             print(len(user_df))
#             # merged_df['prediction'] = predictions

#             output_csv_path = 'predicted_results.csv'
#             user_df.to_csv(output_csv_path, index=False)
            
#             return send_file(output_csv_path, as_attachment=True)
#     return '''
#     <!doctype html>
#     <title>Bulk Predict</title>
#     <h1>Upload CSV file for bulk prediction</h1>
#     <form method=post enctype=multipart/form-data>
#       <input type=file name=file>
#       <input type=submit value=Upload>
#     </form>
#     '''

'''
for testing use file1.csv
'''

@app.route('/bulk-predict', methods=['GET', 'POST'])
def bulk_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            user_df = pd.read_csv(file)
            if 'contract' not in user_df.columns or 'fraud_consumption' not in user_df.columns:
                return 'Invalid file format. The CSV must contain "contract_no" and "fraud_consumption" columns.'

            main_df = pd.read_csv('data_final_v3.csv')

            merged_df = pd.merge(main_df, user_df, how='inner', left_on='contract', right_on='contract')
            print(f'Number of rows in merged_df: {len(merged_df)}')

            merged_df['fraud_consumption'] = merged_df['fraud_consumption_y']

            # Drop duplicates based on the user_df's original rows
            merged_df = merged_df.drop_duplicates(subset=user_df.columns)
            print(f'Number of rows in merged_df after dropping duplicates: {len(merged_df)}')
            required_columns = ['Unnamed: 0','billing_type', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 'fraud_consumption',
                   'SERVICE_STATUS', 'POWER_SUSCRIBED', 'TARIFF', 'ACTIVITY_CMS', 'READWITH', 'SEGMENT',
                   'agency', 'zone','block']
            df1 = merged_df[required_columns]
            imputer = SimpleImputer(strategy='mean')
            df_imputed = pd.DataFrame(imputer.fit_transform(df1), columns=df1.columns)
            print(f'Number of rows in df_imputed: {len(df_imputed)}')
            loaded_svm_model = joblib.load('model_v1.pkl')
            predictions = loaded_svm_model.predict(df_imputed)
            print(f'Number of predictions: {len(predictions)}')
            print(f'Number of rows in user_df: {len(user_df)}')

            if len(predictions) == len(user_df):
                user_df['prediction'] = predictions
            else:
                return 'Error: Mismatch in the number of predictions and user data rows.'

            output_csv_path = 'predicted_results.csv'
            user_df.to_csv(output_csv_path, index=False)
            
            return send_file(output_csv_path, as_attachment=True)
    return '''
    <!doctype html>
    <title>Bulk Predict</title>
    <h1>Upload CSV file for bulk prediction</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)



if __name__ == '__main__':
    app.run(debug=True)
