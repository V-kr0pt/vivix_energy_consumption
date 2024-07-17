import torch
from torch.utils.data import DataLoader, TensorDataset

import mlflow
from mlflow.models import infer_signature

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.model_utils import Model_utils 
from utils.load_data import LoadData 
from utils.preprocess import Preprocess


class LSTM_model:

    def __init__(self, model_params: dict, comments: str):
        self.model_utils = Model_utils()
        self.load_data = LoadData()
        
        self.model = None
        self.model_name = "LSTM"
        self.model_params = model_params
        self.comments = comments
           
    def create_model(self, input_size):
        hidden_layer_size = self.model_params['hidden_layer_size']
        num_layers = self.model_params['num_layers']
        output_size = self.model_params['output_size']
        #seq_length = self.model_params['seq_length']

        self.model = torch.nn.LSTM(input_size=input_size,
                                    hidden_size=hidden_layer_size, 
                                    num_layers=num_layers, batch_first=True)
        
        # Fully connected layer
        # to map the output of the LSTM to the output size    
        self.fc = torch.nn.Linear(hidden_layer_size, output_size) 
    
    def create_Dataloader(self, X, y, batch_size=32):
        # Create DataLoader for batching
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataloader

    def create_sequences(self, X, y):
        seq_length = self.model_params['seq_length']
        X_seq = []
        y_seq = []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        return X_seq, y_seq
    
    def forward_pass(self, x):
        lstm_out, _ = self.model(x)
        out = self.fc(lstm_out)
        return out.view(-1) # to be an array and not a 1 column matrix

    def train_model(self, train_loader):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_params['learning_rate'])

        self.model.train()
        for epoch in range(self.model_params['num_epochs']):
            epoch_loss = 0
            for (X_batch, y_batch) in train_loader:
                optimizer.zero_grad()
                output = self.forward_pass(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f'Epoch: {epoch}, Loss: {epoch_loss}')
    
  
    def calculate_metrics(self, y_test, y_pred):
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse**0.5
        r2 = r2_score(y_test, y_pred)
        return mae, mse, rmse, r2
    
    def run(self):
         # Load data
        data_train = self.load_data.data        
        # Create the preprocess object and the preprocessor
        preprocess = Preprocess(data_train, self.load_data.numerical_features, self.load_data.categorical_features,
                                self.load_data.boolean_features, self.load_data.target)
        preprocessor = preprocess.create_preprocessor(imputer_stategy=None, scale_std=False, scale_minmax=False)
              
        # Features and target split
        features = self.load_data.features
        target = self.load_data.target

        X_train = data_train[features] # the train data is converted to a numpy array after the fit_transform of the preprocessor
        y_train = data_train[target].to_numpy() # Convert to numpy array
        
        # Preprocess the data
        X_train = preprocessor.fit_transform(X_train)        
        
        # Create sequences (should I create this function in model_utils.py ?)
        #X_train, y_train = self.create_sequences(X_train, y_train, self.model_params['seq_length'])
        
        # Create DataLoader
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        train_dataloader = self.create_Dataloader(X_train, y_train)
        
        # Create and train the model
        self.create_model(input_size=X_train.shape[1])
        self.train_model(train_dataloader)

        # Test the model
        data_test = self.load_data.last_month_data

        X_test = data_test[features]
        y_test = data_test[target].to_numpy()

        # preprocess the test data
        X_test = preprocessor.transform(X_test)

        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).float()       
        
        y_pred = self.forward_pass(X_test)
        
        # Transform the predictions to numpy
        y_pred = y_pred.detach().numpy()
        y_test = y_test.detach().numpy()

        # Calculate the metrics
        mae, mse, rmse, r2 = self.calculate_metrics(y_test, y_pred)
        plot_path = self.model_utils.plot_predictions(y_pred, y_test, mae, mse, rmse, r2, self.model_name)
        print(f'MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}')
        
        # Set our tracking server uri for logging
        host = "127.0.0.1"
        port = 8080
        mlflow.set_tracking_uri(uri=f"http://{host}:{port}")

        # Create a new MLflow Experiment
        mlflow.set_experiment("MLflow Vivix")
        with mlflow.start_run():

            # Log metrics
            mlflow.log_metric('MAE', mae)
            mlflow.log_metric('MSE', mse)
            mlflow.log_metric('RMSE', rmse)
            mlflow.log_metric('R2', r2)
            
            # Log hyperparams
            mlflow.log_params(self.model_params)

            # Save the prediction plot
            mlflow.log_artifact(plot_path)
            
            # Set a tag that we can use to remind ourselves what this run was for
            mlflow.set_tag("Training Info", self.comments)

            # Infer the model signature
            signature = infer_signature(X_train.detach().numpy(),
                                         self.forward_pass(X_train).detach().numpy())

            mlflow.pytorch.log_model(self.model,
                                      self.model_name, signature=signature)
        

if __name__ == "__main__":
    comments = 'First run'

    model_params = {'input_size': 1,
                'hidden_layer_size': 100,
                'output_size': 1,
                'num_layers': 1,
                'seq_length': 1,
                'learning_rate': 0.001,
                'num_epochs': 10}
    
    lstm = LSTM_model(model_params, comments)
    lstm.run()