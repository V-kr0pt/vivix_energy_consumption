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
            '%CACO': 'porcentagem_caco'
        }, inplace=True)

        # Standardize string columns to lowercase
        self.data = data.map(lambda x: x.lower() if isinstance(x, str) else x)

        # Identificar colunas categóricas e numéricas
        self.boolean_features = ['prod_e', 'prod_l'] # Colunas booleanas
        self.categorical_features = ['cor']  # Colunas categóricas
        self.numerical_features = ['boosting', 'espessura', 'extracao_forno', 'porcentagem_caco']  # Colunas numéricas
        
        
        self.features = self.numerical_features + self.categorical_features + self.boolean_features  # Colunas de entrada
        self.target = 'Médio diário'  # Coluna alvo



    def create_preprocessor(self, scale_std=False, scale_minmax=False):

        # Transformer to numerical columns
        steps = [('imputer', SimpleImputer(strategy='mean'))] # If there are missing values, fill with the mean

        # Add scaler to the pipeline
        if scale_std and scale_minmax:
            raise ValueError('Only one scaler can be selected')
        elif scale_std:
            steps.append(('scaler', StandardScaler()))
        elif scale_minmax:
            steps.append(('scaler', MinMaxScaler()))

        numeric_transformer = Pipeline(steps=steps) # Pipeline to numerical columns

        
        # Transformer to categorical columns
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder())
        ])


        # Preprocessador
        preprocessor = ColumnTransformer(transformers=[
                ('num', numeric_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features),
                ('bool', 'passthrough', self.boolean_features)])
        

        return preprocessor

