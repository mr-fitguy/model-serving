import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
#model = pickle.load(open('model_pkl-dt.pkl', 'rb'))
with open('model_pkl-dt' , 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    
    #int_features = [int(x) for x in request.form.values()]
    int_features = [ x for x in request.form.values()]

    contract_no = int(int_features[0])
    new_consumption = float(int_features[1])

    #contract_no = 200050059
    #new_consumption = 7000.0
    print(contract_no)
    print(new_consumption)
    
    
    #final_features = [np.array(int_features)]
    rawData = pd.read_csv('datatest.csv',  sep=',',  index_col=False)
    print(rawData)

    #rawData=rawData.loc[(rawData['contract']==200050059)]
    #print(rawData)

    rawData['fraud_flag'] = np.where(rawData['fraud_flag'] == 1.0, 1, 0)
    rawData['fraud_consumption'] = new_consumption
    print("rawData : ",rawData)

    infoData = pd.DataFrame()
    infoData['fraud_flag'] = rawData['fraud_flag']
    infoData['contract'] = rawData['contract']
    data = rawData.drop(['fraud_flag', 'contract'], axis=1)

    dropIndex = data[data.duplicated()].index  # duplicates drop
    data = data.drop(dropIndex, axis=0)
    infoData = infoData.drop(dropIndex, axis=0)

    zeroIndex = data[(data.sum(axis=1) == 0)].index  # zero rows drop
    data = data.drop(zeroIndex, axis=0)
    infoData = infoData.drop(zeroIndex, axis=0)



    data.reset_index(inplace=True, drop=True)  # index sorting
    infoData.reset_index(inplace=True, drop=True)

    data = data.interpolate(method='linear', limit=2,  # filling NaN values
                        limit_direction='both', axis=0).fillna(0)



    for i in range(data.shape[0]):  # outliers treatment
        m = data.loc[i].mean()
        st = data.loc[i].std()
        data.loc[i] = data.loc[i].mask(data.loc[i] > (m + 3 * st), other=m + 3 * st)



    scale = MinMaxScaler()
    scaled = scale.fit_transform(data.values.T).T
    mData = pd.DataFrame(data=scaled, columns=data.columns)
    preprData = pd.concat([infoData, mData], axis=1, sort=False)  # Back to initial format

    preprData = preprData.rename(columns={"1": "1/1/2018"})
    preprData = preprData.rename(columns={"2": "1/2/2018"})
    preprData = preprData.rename(columns={"3": "1/3/2018"})
    preprData = preprData.rename(columns={"4": "1/4/2018"})
    preprData = preprData.rename(columns={"5": "1/5/2018"})
    preprData = preprData.rename(columns={"6": "1/6/2018"})
    preprData = preprData.rename(columns={"7": "1/7/2018"})
    preprData = preprData.rename(columns={"8": "1/8/2018"})
    preprData = preprData.rename(columns={"9": "1/9/2018"})
    preprData = preprData.rename(columns={"10": "1/10/2018"})
    preprData = preprData.rename(columns={"11": "1/11/2018"})
    preprData = preprData.rename(columns={"12": "1/12/2018"})
    preprData = preprData.rename(columns={"fraud_consumption": "1/1/2019"})


    postData =  preprData[["1/1/2018","1/2/2018","1/3/2018","1/4/2018","1/5/2018","1/6/2018","1/7/2018","1/8/2018","1/9/2018","1/10/2018","1/11/2018","1/12/2018","1/1/2019"]]


    int_features = np.array(postData.values)
    final_features = np.array(int_features)

    
    print(final_features)
    prediction = model.predict(final_features)
    print("prediction : ",round(prediction[0], 2))

    
   

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Predication is {}'.format(output))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
