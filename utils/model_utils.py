import pickle
from rich.progress import Progress, SpinnerColumn, TextColumn
from datetime import datetime
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Model_utils:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.model_path = None
        pass

    # Train the machine learning model (not a deep learning model)
    def train_model(self, model, X_train, y_train, model_name, save=True, grid_search=False, param_grid=None, n_splits=3, comments=None):
        
        # comments about the train model to be saved
        self.comments = comments

        # create TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)

        if grid_search:
            # with progress to show a progress bar (rich library)
            with Progress(SpinnerColumn(spinner_name='squish'), 
                          TextColumn('[progress.description]{task.description}')) as progress:
                
                task = progress.add_task("[purple4]Grid Search in progress...", total=None)
                
                # Grid search to find the best parameters
                grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, n_jobs=-1, verbose=2)
                grid_search.fit(X_train, y_train)
                self.best_params = grid_search.best_params_
                self.best_score = grid_search.best_score_
                model = model.set_params(**grid_search.best_params_)
                
                # Updating the task to indicate it is complete
                progress.update(task, description="[green]Grid Search Done!")
        
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
    def load_model(self, model_path=None):
        
        # If the model path is not provided, use the model path that was saved
        if self.model_path is None and model_path is None:
            raise ValueError('model_path is None. You need to provide a model path to load the model')
        elif model_path is not None:
            self.model_path = model_path

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
                file.write('model_name,date,MAE,MSE,RMSE,R2,model_params,comments\n') 

        # Append the results
        with open(f'{folder}/{file_name}', 'a') as file:            
            file.write(f'{self.model_name},{now},{mae},{mse},{rmse},{r2},\"{self.model.get_params()}\",\"{self.comments}\"\n')


    def plot_predictions(self, y_true, y_pred, graph_name='prediction', save=True):

        # Calcular o RMSE
        rmse_value = np.sqrt(mean_squared_error(y_true, y_pred))

        # Create the error bands
        upper_band = y_pred + rmse_value
        lower_band = y_pred - rmse_value

        # create theme
        sns.set_theme(style="darkgrid")
        
        # Create a figure and axis  
        fig, ax = plt.subplots(figsize=(15, 7))

        # Plot the true values
        sns.lineplot(x=range(len(y_true)), y=y_true, label='True Values', ax=ax)

        # Plot the predicted values
        sns.lineplot(x=range(len(y_pred)), y=y_pred, label='Predicted Values', ax=ax)

        # Calculate the RMSE
        rmse_value = np.sqrt(mean_squared_error(y_true, y_pred))

        # Create the error bands
        upper_band = y_pred + rmse_value
        lower_band = y_pred - rmse_value

        # Plot the error bands
        ax.fill_between(range(len(y_pred)), lower_band, upper_band, alpha=0.3, label='Error Band', color='yellow')

        # Set the title
        ax.set_title('Consumo de energia médio diário do forno')
        # Set the y label
        ax.set_ylabel('Consumo de energia (MWh/dia)')

        # Set the legend
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        if save:
            save_graph_path = 'results/graphs/'
            # Save the plot
            fig.savefig(save_graph_path + graph_name + '.png')