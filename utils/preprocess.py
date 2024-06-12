import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

class LoadData:
    def __init__(self):
         # Load data
        data = pd.read_excel('content/data_trainE_L_MARÇO_COMPLETO.xlsx')

        # Rename columns
        data.rename(columns={
            'BOOSTING (MWH)': 'boosting',
            'Cor': 'cor',
            'Prod_E': 'prod_e',
            'Prod_L': 'prod_l',
            'Espess.': 'espessura',
            'Extração forno': 'extracao_forno',
            '%CACO': 'porcentagem_caco',
            'Médio diário': 'medio_diario'
        }, inplace=True)

        # Standardize string columns to lowercase
        self.data = data.map(lambda x: x.lower() if isinstance(x, str) else x)

        # Identificar colunas categóricas e numéricas
        self.boolean_features = ['prod_e', 'prod_l'] # Colunas booleanas
        self.categorical_features = ['cor']  # Colunas categóricas
        self.numerical_features = ['boosting', 'espessura', 'extracao_forno', 'porcentagem_caco']  # Colunas numéricas
        
        self.features = self.numerical_features + self.categorical_features + self.boolean_features  # Input columns
        self.target = 'medio_diario'  # Target column

    def create_lag_columns(self, lag_columns, lag_values):
        # Create lag columns
        for column, value in zip(lag_columns, lag_values):
            lag_column_name = column+f'_lag{value}'
            self.data[lag_column_name] = self.data[column].shift(value)

            if column in self.numerical_features or column == self.target:
                self.numerical_features.append(lag_column_name)
            elif column in self.categorical_features:
                self.categorical_features.append(lag_column_name)
            elif column in self.boolean_features:
                self.boolean_features.append(lag_column_name)
        
        # update features
        self.features = self.numerical_features + self.categorical_features + self.boolean_features 

        return self.data 

    def create_preprocessor(self, imputer_stategy=None, scale_std=False, scale_minmax=False):

        # Transformer to numerical columns
        if imputer_stategy is None:
            step = []
        else:
            step = [('imputer', SimpleImputer(strategy='mean'))] # If there are missing values, fill with the mean (Maybe change because of the lag columns)

        # Add scaler to the pipeline
        if scale_std and scale_minmax:
            raise ValueError('Only one scaler can be selected')
        elif scale_std:
            step.append(('scaler', StandardScaler()))
        elif scale_minmax:
            step.append(('scaler', MinMaxScaler()))

        if step != []:
            numeric_transformer = Pipeline(steps=step) # Pipeline to numerical columns
        else:
            numeric_transformer = 'passthrough'

        # Transformer to categorical columns
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder())])

        # Transformers pipeline
        transformers = [
            ('num', numeric_transformer, self.numerical_features),
            ('cat', categorical_transformer, self.categorical_features),
            ('bool', 'passthrough', self.boolean_features)
        ]

        # Create the preprocessor
        preprocessor = ColumnTransformer(transformers=transformers)

        return preprocessor