import joblib
import pandas as pd
from utils.preprocess import Preprocess
import numpy as np

class Prediction:
    def __init__(self, model_name, data_name, energy_recurrence=False, probability_prediction=False):
        self.model_name = model_name
        self.data_name = data_name
        self.energy_recurrence = energy_recurrence
        self.probability_prediction = probability_prediction
        self.prediction_column_name = 'predicted_consumo'
        
    def load_model(self):
        path = './results/models/' + self.model_name + '.pkl'
        self.model = joblib.load(path)
        return 

    def load_data(self):
        data_path = './results/input_data/' + self.data_name + '.xlsx'
        self.data = pd.read_excel(data_path)

    def preprocess_data(self):
        # Perform data preprocessing here
        self.data.rename(columns={
            'data':'datetime',
            'cor': 'cor',
            'esp': 'espessura',
            'extração': 'extracao_forno',
            'caco': 'porcentagem_caco',
            'ext forno boosting': 'extracao_boosting',
            'potencia boosting': 'boosting'
        }, inplace=True) 

        # if recurrent prediction we have to change another column name
        if self.energy_recurrence:
            self.data.rename(columns={
                'medio_diario': 'consumo_medio_diario'
            }, inplace=True)
        
        # we'll return the 'cor' to their orignal names, so we can use the same preprocessing as before
        cor_dict = {0:'incolor', 1:'verde', 2:'cinza'}
        self.data['cor'] = self.data['cor'].map(cor_dict)

        # be sure that all strings are lower case (cor is already lowercase)
        #self.data = self.data.applymap(lambda x: x.lower() if isinstance(x, str) else x)

        # list of features
        self.numerical_features = ['boosting', 'espessura', 'extracao_forno', 'porcentagem_caco']
        self.categorical_features = ['month', 'week_day', 'day', 'cor']
        self.boolean_features = []

        self.features = self.numerical_features + self.categorical_features + self.boolean_features

        # target
        self.target = 'consumo_medio_diario'

        # Create a Preprocess object
        preprocess = Preprocess(self.numerical_features, self.categorical_features,
                                 self.boolean_features, )
        
        #preprocessor = preprocess.create_preprocessor(imputer_stategy=None, scale_std=False, scale_minmax=False)
        
        # preprocess the data
        # transform the boosting column
        self.data['boosting'] = self.data.apply(
            lambda row: preprocess.boost_power(row['cor'], row['extracao_boosting']), axis=1
        )

        # create date columns
        self.data['datetime'] = pd.to_datetime(self.data['datetime'], dayfirst=True)
        self.data['month'] = self.data['datetime'].dt.month
        self.data['week_day'] = self.data['datetime'].dt.dayofweek
        self.data['day'] = self.data['datetime'].dt.day

        # create lag target column
        if self.energy_recurrence:
            lag_columns_list = [self.target]*7
            lag_values = [1, 2, 3, 4, 5, 6, 7]
            self.data = preprocess.create_lag_columns(lag_columns_list, lag_values)
            self.data = self.data.iloc[7:]

        # Tranform the columns to use OneHotEncoder
        preprocessor_path = './results/preprocessors/' + self.model_name + '_preprocessor.pkl'
        preprocess.load_preprocessor(preprocessor_path)
        
        # Remove target_data and datetime columns
        if self.energy_recurrence:
            self.target_data = self.data[self.target].values.reshape(-1,1)
            self.data.drop(columns=[self.target], inplace=True)
        self.datetime_data = self.data['datetime'].values.reshape(-1,1)
        self.data.drop(columns=['datetime'], inplace=True)

        # Transform the data and concatenate the target_data
        self.data = preprocess.transform(self.data) 
        #self.data = np.hstack([self.data, self.target_data])
        
        self.features = preprocess.features
        self.target_column_index = len(self.features) # the target is the last column
        
        
    def recurrent_prediction(self):
        # Inicializar as previsões
        predictions = []

        for i, row in enumerate(self.data):
            prediction = self.model.predict(row.reshape(1,-1))[0] # transform to 2D array and do the prediction

            predictions.append({
                'datetime': self.datetime_data[i,0],
                'predicted_consumo': prediction
            })
            
            # update all the lag columns
            # the lag columns are the 5th column to 11th column (self.features[4] = lag1 and self.features[10] = lag7)
            for j in range(0, 7):
                upt_col = j+4 # the first lag column is the 5th column
                upt_row = i+j+1 # the row to be updated
                if i+j+1 >= len(self.data):
                    break
                else:
                    self.data[upt_row, upt_col] = prediction
        
        return predictions

    def prediction(self):
        predictions = []

        prediction = self.model.predict(self.data)
     
        if self.probability_prediction:
            self.prediction_column_name = 'estouro_previsto'
            prob_prediction = self.model.predict_proba(self.data)[:,1]
            
            
        for i, row in enumerate(prediction):
            dict_predictions = {}
            dict_predictions['datetime'] = self.datetime_data[i,0]
            dict_predictions[self.prediction_column_name] = row
            if self.probability_prediction:
                dict_predictions['probabilidade_de_estouro'] = prob_prediction[i]
            
            predictions.append(dict_predictions)
        
        
        return predictions

    def run(self):
        # load the model and the data
        self.load_model()
        self.load_data()
        # preprocess the data
        self.preprocess_data()

        # make predictions
        if self.energy_recurrence:
            predictions = self.recurrent_prediction()
        else: 
            predictions = self.prediction()

        # Create a DataFrame with the predictions
        if self.probability_prediction:
            predictions_df = pd.DataFrame(predictions, columns=['datetime', self.prediction_column_name, 'probabilidade_de_estouro'])
        else:
            predictions_df = pd.DataFrame(predictions, columns=['datetime', self.prediction_column_name])
        
        # saving predictions
        predictions_df.to_csv('./results/output_data/' + self.model_name + '.csv', index=False)

if __name__ == '__main__':
    model_name = 'xgboost_2024-09-11_09-21-24'
    data_name = 'prediction_data_prod_E_L'
    prediction = Prediction(model_name, data_name, probability_prediction=True)
    prediction.run()