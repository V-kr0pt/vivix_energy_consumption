import pandas as pd

def adjust_time(row):
        if row['hora'] == 24:
            return row['data'] + pd.Timedelta(days=1)
        else:
            return pd.to_datetime(f"{row['data']}") 



def preprocess_data(data):
    data.columns = data.columns.str.lower()

    # transform kWh to MWh
    data['ativa c (kwh)'] = data['ativa c (kwh)'].astype('str').str.replace('.', '')
    data['ativa c (kwh)'] = data['ativa c (kwh)'].astype('str').str.replace(',', '.')
    data['consumo_mwh'] = data['ativa c (kwh)'].astype('float') / 1000
        
    # create datetime column to group by and sum the consumption
    data['data'] = pd.to_datetime(data['data'], format='%d/%m/%Y')
    data['datetime'] = data.apply(adjust_time, axis=1)

    # Sum the energy consumption from differents ponto_de_medicao grouping by datetime 
    data = data.groupby(['datetime'])['consumo_mwh'].sum().reset_index()

    # Assure that the data is sorted by date
    data = data.sort_values(by='datetime')

    # Reset the index
    data.reset_index(drop=True, inplace=True)

    return data

if __name__ == '__main__':
    path = 'VIVIX_CARGA_2024.csv'
    data = pd.read_csv(path)
    data = preprocess_data(data)
    data.to_csv('ground_truth.csv', index=False)