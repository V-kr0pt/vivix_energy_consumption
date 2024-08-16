import pandas as pd

class LoadData:
    def __init__(self, path='content/potencias_geral.xlsx'):
        data = pd.read_excel(path)
        old_data = pd.read_excel('content/cleaned_data_train.xlsx')

        # Rename columns
        # all in lower case
        data.columns = data.columns.str.lower()

        # transform kWh to MWh
        data['ativa consumo (kwh)'] = data['ativa consumo (kwh)'] / 1000
        data.rename(columns={'ativa consumo (kwh)': 'consumo_mwh'}, inplace=True) 

        # Rename columns
        old_data.rename(columns={
            'BOOSTING (MWH)': 'boosting',
            'Cor': 'cor',
            'Prod_E': 'prod_e',
            'Prod_L': 'prod_l',
            'Espess.': 'espessura',
            'Extração forno': 'extracao_forno',
            '%CACO': 'porcentagem_caco',
            'Médio diário': 'medio_diario',
            'Data':'datetime'
        }, inplace=True)       

        # create datetime column to group by and sum the consumption
        data['datetime'] = data.apply(self.adjust_time, axis=1)

        # Group by datetime and ponto_de_medicao
        data = data.groupby(['datetime'])['consumo_mwh'].sum().reset_index()

        # Standardize string columns to lowercase
        data = data.applymap(lambda x: x.lower() if isinstance(x, str) else x)

        # Garantee that the data is sorted by date
        data = data.sort_values(by='datetime')

        # now divide the data column into year, month, week_day and hour
        data['year'] = data['datetime'].dt.year
        data['month'] = data['datetime'].dt.month
        data['day'] = data['datetime'].dt.day
        data['week_day'] = data['datetime'].dt.dayofweek
        data['hour'] = data['datetime'].dt.hour

        # Now we only need the max consumation value from the day
        data['consumo_max_diario'] = data.groupby(['year', 'month', 'day'])['consumo_mwh'].transform('max')
        data = data[data['consumo_mwh'] == data['consumo_max_diario']]
        data.drop(columns=['consumo_mwh'], inplace=True)

        # we can concatenate the old data with the new one
        # unfortunnaly we will losing some data but we have more features
        print('Data lost: ', data.shape[0] - old_data.shape[0])
        data['datetime'] = data['datetime'].dt.date
        old_data['datetime'] = old_data['datetime'].dt.date
        data = pd.merge(data, old_data, on='datetime')
        # the medio_diario is not useful anymore
        data.drop(columns=['medio_diario'], inplace=True)

        ## we'll create binary classes for the consumption using 8.3 mWh as threshold
        data['crossed_threshold'] = data['consumo_max_diario'].apply(lambda x: 1 if x >= 8.3 else 0)

        # Create a different dataframe with only the last month data (last 30 days)
        last_30_days = data['datetime'].max() - pd.Timedelta(days=30)
        self.last_month_data = data.loc[data['datetime'] >= last_30_days].copy()
        self.data = data.loc[data['datetime'] < last_30_days].copy()

        # Finally we can drop the datetime column
        self.data.drop(columns=['datetime'], inplace=True)
        self.last_month_data.drop(columns=['datetime'], inplace=True)

        # Identify categorical and numerical columns
        self.boolean_features = ['prod_e', 'prod_l'] # Boolean columns
        self.categorical_features = ['year', 'month', 'week_day', 'hour', 'day', 'cor']  # Categorical columns
        self.numerical_features = ['boosting', 'espessura', 'extracao_forno', 'porcentagem_caco']   # Numerical columns
        
        # create a list with all features and the target
        self.features = self.numerical_features + self.categorical_features + self.boolean_features
        self.target = 'crossed_threshold'


    def adjust_time(self, row):
        if row['hora'] == 24:
            return row['data'] + pd.Timedelta(days=1)
        else:
            return pd.to_datetime(f"{row['data']} {row['hora']}:00") 


if __name__ == '__main__':
    load_data = LoadData()
    print('\n\t\t---- Data ----')
    print(load_data.data.head())
    print('shape: ', load_data.data.shape)
    print('\n\t\t---- Last month data ----\t')
    print(load_data.last_month_data.head())
    print('shape: ', load_data.last_month_data.shape)
    print('\n\t\t---- Features ----')
    print(load_data.features)
    print('number of features: ', len(load_data.features))
    print('\n\t\t---- Target ----')
    print(load_data.target)
    print('\n')
