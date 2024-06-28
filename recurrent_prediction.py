import pandas as pd
import numpy as np
from datetime import timedelta
from utils.preprocess import LoadData 
from utils.model_utils import Model_utils 


class Model:

    def __init__(self, model_name, model_date):
        model_utils = Model_utils()
        model_path=f'models/{model_name}__{model_date}.pkl'
        preprocessor_path = f'models/preprocessors/{model_name}__{model_date}_preprocessor.pkl'

        self.model, self.preprocessor = model_utils.load_model(model_path, preprocessor_path)

    # after should be a sklearn pipeline
    def data_preprocess(self, data, rename_columns_dict=None):
        self.preprocessed_data = data
        load_data = LoadData(new_data=True) # LoadData class instance without load the training data        

        if rename_columns_dict is not None:
            self.preprocessed_data = self.preprocessed_data.rename(columns=rename_columns_dict)

        # Ordering by date
        data = self.preprocessed_data.sort_values(by='date')
        
        # Standardize string columns to lowercase
        self.preprocessed_data = self.preprocessed_data.map(lambda x: x.lower() if isinstance(x, str) else x)
        
        # add week_day column
        self.preprocessed_data['week_day'] = self.preprocessed_data['date'].dt.dayofweek

        # lagging columns
        lag_columns_list = ['medio_diario']*7
        lag_values = [1, 2, 3, 4, 5, 6, 7]

        # create the lagged columns in data
        self.preprocessed_data = load_data.create_lag_columns(self.preprocessed_data, lag_columns_list, lag_values)
        self.preprocessed_data = self.preprocessed_data.iloc[7:]
        
        # preprocess feature columns and add target without preprocessing
        self.preprocessed_data = self.preprocessor.transform(self.preprocessed_data[load_data.features])
        
        # NOW IS A NUMPY ARRAY WITHOUT COLUMNS NAMES!       
        target_data = data[load_data.target].to_numpy().reshape(-1, 1)[7:]
        self.preprocessed_data = np.concatenate([self.preprocessed_data, target_data], axis=1)       
        
    def prediction(self, prediction_date):
        # Transforming the prediction_date to datetime
        target_date = pd.to_datetime(prediction_date) #pd.to_datetime('2023-12-31')

        # copy the preprocessed data
        data = self.preprocessed_data.copy()

        # While the last data is before the target_date, do the predictions
        while data['date'].max() < target_date:
            # Selecting the last row since it's the more recent data
            last_observation = data.iloc[-1:]

            # Doing the prediction using the last observation
            X = last_observation[self.feature_list]
            y_pred = self.model.predict(X)

            # Crie uma nova linha com a previsão e adicione ao DataFrame
            # Aqui você precisa ajustar conforme a estrutura do seu DataFrame
            new_line = last_observation.copy()
            new_line[self.target] = y_pred  # Saving the prediction in the target column
            new_line['date'] = new_line['date'] + timedelta(days=1)  # Updating the date
            
            # Atualize as colunas lagged com os valores corretos
            # Este passo depende de como você implementou a criação das colunas lagged
            for i in range(1, 8):
                new_line[f'{self.target}_lag_{i}'] = data[self.target].shift(i)

            # Atualize as colunas lagged com os valores corretos
            # Este passo depende de como você implementou a criação das colunas lagged

            data = pd.concat([data, new_line])

        return data


if __name__ == '__main__':

    #load data excel
    data = pd.read_excel('./content/cleaned_data_train.xlsx')

    # load the model and preprocessor
    model = Model('xgboost', '2024-06-26_21-06-24')
    
    # Rename columns
    rename_columns_dict = {
        'Data': 'date',
        'BOOSTING (MWH)': 'boosting',
        'Cor': 'cor',
        'Prod_E': 'prod_e',
        'Prod_L': 'prod_l',
        'Espess.': 'espessura',
        'Extração forno': 'extracao_forno',
        '%CACO': 'porcentagem_caco',
        'Médio diário': 'medio_diario'
    }    


    model.data_preprocess(data=data,rename_columns_dict=rename_columns_dict)
    
    results_path = './results/'
    new_data = model.prediction('2024-12-10')
    new_data.to_csv(results_path+'new_data.csv', index=False)