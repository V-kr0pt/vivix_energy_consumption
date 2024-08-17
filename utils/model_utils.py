import numpy as np
import pickle
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, roc_curve, auc, precision_recall_curve

import seaborn as sns
import matplotlib.pyplot as plt

class Model_utils:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.model_path = None
        self.preprocessor_path = None

    # Train the machine learning model (not a deep learning model)
    def train_model(self, model, X_train, y_train, model_name, preprocessor=None, save=True, grid_search=False, param_grid=None, cv=3, comments=None):
        
        # comments about the train model to be saved
        self.comments = comments

        if grid_search:

            # Grid search to find the best parameters
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2)
            grid_search.fit(X_train, y_train)
            self.best_params = grid_search.best_params_
            self.best_score = grid_search.best_score_
            model = model.set_params(**grid_search.best_params_)
        
        model.fit(X_train, y_train)

        if save:
            if preprocessor is None:
                raise ValueError('You should save the preprocessor to save the model and preprocess the data after loading it!')                
            self.model = model
            self.model_name = model_name
            self.preprocessor = preprocessor
            self.save_model()
    
    # Save the model in a pickle file
    def save_model(self):
        now = datetime.now()
        now = now.strftime("%Y-%m-%d_%H-%M-%S")

        folder = 'models'
        file_name = f'{self.model_name}__{now}.pkl'
        preprocessor_folder = folder + '/preprocessors'
        preprocessor_name = f'{self.model_name}__{now}_preprocessor.pkl'

        self.model_path = f'{folder}/{file_name}'
        self.preprocessor_path = f'{preprocessor_folder}/{preprocessor_name}'

        with open(self.model_path, 'wb') as file:
            pickle.dump(self.model, file)

        with open(self.preprocessor_path, 'wb') as file:
            pickle.dump(self.preprocessor, file)

    # Load the model from a pickle file
    def load_model(self, model_path=None, preprocessor_path=None):
        
        # If the model path is not provided, use the model path that was saved (can be None)
        if self.model_path is None and model_path is None:
            raise ValueError('model_path is None. You need to provide a model path to load the model')
        elif model_path is not None:
            self.model_path = model_path

        # If the preprocessor path is not provided, use the preprocessor path that was saved (can be None)
        if self.preprocessor_path is None and preprocessor_path is None:
            print('preprocessor_path is None. You need to provide a preprocessor_path path to load the preprocessor')
        elif preprocessor_path is not None:
            self.preprocessor_path = preprocessor_path

        
        # load the model
        with open(self.model_path, 'rb') as file:
            self.model = pickle.load(file)

        # load the preprocessor
        if self.preprocessor_path is not None:
            with open(self.preprocessor_path, 'rb') as file:
                self.preprocessor = pickle.load(file)

            return self.model, self.preprocessor
        
        else:
            return self.model            

    # Test the model and save the metrics in a csv file 
    def test_model(self, X_test, y_test, save_metrics=True, return_error_metrics=False):
        y_pred = self.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)
        
        
        if save_metrics:
            self.save_csv_results(mae, mse, rmse, r2)

        if return_error_metrics:
            return y_pred, mae, mse, rmse, r2
        else:
            return y_pred
    
    def save_csv_results(self, mae, mse, rmse, r2):
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


    def plot_predictions(self, y_pred, y_true, mae, mse, rmse, r2, model_name,graph_name='prediction',
                          graph_title='Consumo máximo de energia diária do Forno',
                          graph_ylabel='Consumo de energia (MWh/dia)',
                          save=True, save_path='results/graphs/', print_error=False):
        
        # Create the error bands
        upper_band = y_pred + rmse
        lower_band = y_pred - rmse

        # create theme
        sns.set_theme(style="darkgrid")
        
        # Create a figure and axis  
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot the true values
        sns.lineplot(x=range(len(y_true)), y=y_true, label='True Values', ax=ax)

        # Plot the predicted values
        sns.lineplot(x=range(len(y_pred)), y=y_pred, label='Predicted Values', ax=ax)

        # Plot the error bands
        ax.fill_between(range(len(y_pred)), lower_band, upper_band, alpha=0.3, label='Error Band', color='yellow')

        # Set the title
        ax.set_title(graph_title)
        # Set the y label
        ax.set_ylabel(graph_ylabel)

        # Set the legend
        #ax.legend(loc='center left', bbox_to_anchor=(0.7, 0.2))

        # print error metrics
        if print_error:
            print(f'MAE:  {mae}')
            print(f'MSE:  {mse}')
            print(f'RMSE: {rmse}')
            print(f'R2:   {r2}')

        plot_path = save_path + model_name + '_' + graph_name + '.png'
        if save:
            # Save the plot
            fig.savefig(plot_path)

        return plot_path

    def create_sequences(self, X, y, seq_length):
        X_seq = []
        y_seq = []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        return X_seq, y_seq

    def calculate_metrics(self, y_test, y_pred):
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse**0.5
        r2 = r2_score(y_test, y_pred)
        return mae, mse, rmse, r2
    
    def plot_error_by_epoch(self, train_losses, model_name):
        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=range(len(train_losses)), y=train_losses, ax=ax)
        ax.set_title('Error by epoch')
        ax.set_ylabel('Error')
        ax.set_xlabel('Epoch')
        plot_path = f'results/graphs/{model_name}_error_by_epoch.png'
        fig.savefig(plot_path)
        return plot_path
    

    def plot_confusion_matrix(self, y_test, y_pred, model_name):
        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(figsize=(10, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_title('Confusion Matrix')
        plot_path = f'results/graphs/{model_name}_confusion_matrix.png'
        fig.savefig(plot_path)
        return plot_path
    
    def plot_roc(self, y_test, y_pred, model_name):
        fpr, tpr, threshold = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # best threshold
        gmeans = np.sqrt(tpr * (1 - fpr))
        best_idx = np.argmax(gmeans)
        best_threshold = threshold[best_idx]

        line_x = [0, 1] # diagonal line
        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=fpr, y=tpr, ax=ax)
        sns.lineplot(x=line_x, y=line_x, ax=ax)  # plot the diagonal line

        #ax.plot(fpr[best_idx], tpr[best_idx], 'ro', label=f'{best_threshold:.2f}')  # plot the best threshold
        #ax.legend(loc='best')
        
        ax.set_title('ROC Curve')
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')
        plot_path = f'results/graphs/{model_name}_roc_curve.png'
        fig.savefig(plot_path)
        return plot_path, roc_auc, best_threshold
    
    def plot_precision_recall(self, y_test, y_pred, model_name):
        precision, recall, threshold = precision_recall_curve(y_test, y_pred)
        pr_auc = auc(recall, precision)

        f1_score = 2 * (precision * recall) / (precision + recall)
        best_idx = np.argmax(f1_score)
        best_threshold = threshold[best_idx]

        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=recall, y=precision, ax=ax)
        #ax.plot(recall[best_idx], precision[best_idx], 'ro', label=f'{best_threshold:.2f}')  # plot the best threshold
        #ax.legend(loc='best')

        ax.set_title('Precision-Recall Curve')
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        plot_path = f'results/graphs/{model_name}_precision_recall_curve.png'
        fig.savefig(plot_path)
        return plot_path, pr_auc, best_threshold