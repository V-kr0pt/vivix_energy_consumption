import pandas as pd
import numpy as np
from datetime import timedelta
#from utils.preprocess import LoadData 
from utils.model_utils import Model_utils 


class Model:

    def __init__(self, model_name, model_date):
        # Load the model and preprocessor
        model_utils = Model_utils()
        model_path=f'models/{model_name}__{model_date}.pkl'
        preprocessor_path = f'models/preprocessors/{model_name}__{model_date}_preprocessor.pkl'

        self.model, self.preprocessor = model_utils.load_model(model_path, preprocessor_path)

        # Identificar colunas categóricas e numéricas
        self.boolean_features = ['prod_e', 'prod_l'] # Colunas booleanas
        self.categorical_features = ['cor', 'week_day']  # Colunas categóricas
        self.numerical_features = ['boosting', 'espessura', 'extracao_forno', 'porcentagem_caco']  # Colunas numéricas
        
        self.features = self.numerical_features + self.categorical_features + self.boolean_features  # Input columns
        self.target = 'medio_diario'  # Target column


    def create_lag_columns(self, lag_data, lag_columns, lag_values):
        # Create lag columns
        for column, value in zip(lag_columns, lag_values):
            lag_column_name = column+f'_lag{value}'
            lag_data[lag_column_name] = lag_data[column].shift(value)

            if column in self.numerical_features or column == self.target:
                self.numerical_features.append(lag_column_name)
            elif column in self.categorical_features:
                self.categorical_features.append(lag_column_name)
            elif column in self.boolean_features:
                self.boolean_features.append(lag_column_name)
        
        # update features
        self.features = self.numerical_features + self.categorical_features + self.boolean_features 

        return lag_data

    # after should be a sklearn pipeline
    def data_preprocess(self, data, rename_columns_dict=None):
        
        preprocessed_data = data.copy()  

        # renaming columns
        if rename_columns_dict is not None:
            preprocessed_data = preprocessed_data.rename(columns=rename_columns_dict)

        # Ordering by date
        preprocessed_data = preprocessed_data.sort_values(by='date')

        # Keep rows with NaN values in 'medio_diario' column and the 7 rows before it
        null_rows = preprocessed_data['medio_diario'].isnull()
        null_rows = preprocessed_data[null_rows].index
        lower_bound = null_rows[0] - 7
        preprocessed_data = preprocessed_data[lower_bound:]
        
        # Standardize string columns to lowercase
        preprocessed_data = preprocessed_data.map(lambda x: x.lower() if isinstance(x, str) else x)
        
        # add week_day column
        preprocessed_data['week_day'] = preprocessed_data['date'].dt.dayofweek

        # lagging column
        lag_columns_list = ['medio_diario']*7
        lag_values = [1, 2, 3, 4, 5, 6, 7]

        # create the lagged columns in data
        preprocessed_data = self.create_lag_columns(preprocessed_data, lag_columns_list, lag_values)
        preprocessed_data = preprocessed_data.iloc[7:]
        
        # preprocess feature columns and add target without preprocessing
        preprocessed_array = self.preprocessor.transform(preprocessed_data[self.features]) # Now it's an array!
        self.preprocessed_df = pd.DataFrame(preprocessed_array, columns=self.preprocessor.get_feature_names_out()) # Reconstruction of the dataframe
               
        # Obtaining the date column again  
        self.preprocessed_df['date'] = preprocessed_data['date'].values
        
        
    def prediction(self, prediction_date):
        # Transforming the prediction_date to datetime
        target_date = pd.to_datetime(prediction_date) #pd.to_datetime('2023-12-31')

        # copy the preprocessed data
        data = self.preprocessed_df.copy()

        # While the last data is before the target_date, do the predictions
        while self.data['date'].max() < target_date:
            # Selecting the last row since it's the more recent data
            last_observation = data.iloc[-1]
            last_observation = last_observation.drop(['medio_diario','date'])  # Removing the target column

            # Doing the prediction using the last observation
            X = last_observation.values.reshape(1, -1)
            y_pred = self.model.predict(X)


            new_line = last_observation.copy()
            new_line['medio_diario'] = y_pred
            self.last_date = self.last_date + timedelta(days=1)  # Updating the date
            
            # Atualize as colunas lagged com os valores corretos
            # Este passo depende de como você implementou a criação das colunas lagged
            for i in range(1, 8):
                new_line[f'num__medio_diario_lag{i}'] = data['medio_diario'].iloc[-i]

            # Atualize as colunas lagged com os valores corretos
            # Este passo depende de como você implementou a criação das colunas lagged

            data = pd.concat([data, new_line])

        return data


if __name__ == '__main__':

    #load data excel
    data = pd.read_excel('./content/cleaned_recurrent_data_train.xlsx')

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
    new_data = model.prediction('2024-03-20')
    new_data.to_csv(results_path+'new_data.csv', index=False)