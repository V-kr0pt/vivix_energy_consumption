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
        
        
    def prediction(self):
        # copy the preprocessed data
        data = self.preprocessed_df.copy()
        data['medio_diario'] = None
        max_rows = len(data)-1

        # While the last data is before the target_date, do the predictions
        for i, row in data.iterrows():
            # Selecting the last row since it's the more recent data
            row = row.drop(['medio_diario','date'])  # Removing the target column

            # Doing the prediction using the last observation
            X = row.values.reshape(1, -1)
            y_pred = self.model.predict(X)

            data.loc[i,'medio_diario'] = y_pred[0]  # Updating the target column

            for j in range(1, 8):
                data.loc[i+j, f'num__medio_diario_lag{j}'] = y_pred[0]
                if i+j >= max_rows:
                    break

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
    new_data = model.prediction()
    new_data.to_csv(results_path+'recurrent_data_pred.csv', index=False)