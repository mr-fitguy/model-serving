from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
import numpy as np

app = Flask(__name__)
loaded_svm_model = joblib.load('final_svm_model.pkl')
required_columns = ['billing_type', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '1/1/2019',
                   'SERVICE_STATUS', 'POWER_SUSCRIBED', 'TARIFF', 'ACTIVITY_CMS', 'READWITH', 'SEGMENT',
                   'agency', 'zone']

@app.route('/')
def index():
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
