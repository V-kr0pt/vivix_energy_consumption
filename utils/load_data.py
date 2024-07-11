import pandas as pd

class LoadData:
    def __init__(self, path='content/cleaned_data_train.xlsx'):
        data = pd.read_excel(path)

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
        data = data.applymap(lambda x: x.lower() if isinstance(x, str) else x)

        # Add week_day column
        data['week_day'] = data['Data'].dt.dayofweek

        # Create a different dataframe with only the last month data
        last_month = data['Data'].max()
        self.last_month_data = data[data['Data'] >= last_month.replace(day=1)]
        self.data = data[data['Data'] < last_month.replace(day=1)]

        # Identify categorical and numerical columns
        self.boolean_features = ['prod_e', 'prod_l']  # Boolean columns
        self.categorical_features = ['cor', 'week_day']  # Categorical columns
        self.numerical_features = ['boosting', 'espessura', 'extracao_forno', 'porcentagem_caco']  # Numerical columns
        
        self.features = self.numerical_features + self.categorical_features + self.boolean_features
        self.target = 'medio_diario'
