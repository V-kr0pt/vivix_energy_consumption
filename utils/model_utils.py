import pickle
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class Model_utils:
    def __init__(self):
        self.model = None
        self.model_name = None
        pass

    # Train the machine learning model (not a deep learning model)
    def train_model(self, model, X_train, y_train, model_name, save=True):
        model.fit(X_train, y_train)

        if save:
            self.model = model
            self.model_name = model_name
            self.save_model()
    
    # Save the model in a pickle file
    def save_model(self):
        now = datetime.now()
        now = now.strftime("%Y-%m-%d_%H-%M-%S")

        folder = 'models'
        file_name = f'{self.model_name}__{now}.pkl'

        self.model_path = f'{folder}/{file_name}'

        with open(self.model_path, 'wb') as file:
            pickle.dump(self.model, file)

    # Load the model from a pickle file
    def load_model(self):
        with open(self.model_path, 'rb') as file:
            self.model = pickle.load(file)
        return self.model

    # Test the model and save the metrics in a csv file 
    def test_model(self, X_test, y_test, save_metrics=True):
        y_pred = self.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = mse ** 0.5
        
        if save_metrics:
            self.save_csv_results(mae, mse, r2, rmse)

        return y_pred
    
    def save_csv_results(self, mae, mse, r2, rmse):
        now = datetime.now()
        now = now.strftime("%Y-%m-%d_%H-%M-%S")

        folder = 'results'
        file_name = 'history.csv'

        # if the file does not exist, create the header
        try:
            with open(f'{folder}/{file_name}', 'r') as file:
                pass
        except FileNotFoundError:
            with open(f'{folder }/{file_name}', 'w') as file:
                file.write('model_name,date,MAE,MSE,RMSE,R2\n')

        # Append the results
        with open(f'{folder}/{file_name}', 'a') as file:            
            file.write(f'{self.model_name},{now},{mae},{mse},{rmse},{r2}\n')



    


    



