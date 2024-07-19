import mlflow
from mlflow.models import infer_signature

from utils.preprocess import Preprocess
from utils.load_data import LoadData 
from utils.model_utils import Model_utils 

from LSTM_model import LSTMModelWrapper as LSTMModel


comments = ''

params = {'hidden_layer_size': 100,
          'output_size': 1,
          'num_layers': 1,
          'learning_rate': 0.001,
          'epochs': 10}


# Load data
load_data = LoadData()
data = load_data.data        
# Create the preprocess object and the preprocessor
preprocess = Preprocess(data, load_data.numerical_features, load_data.categorical_features,
                        load_data.boolean_features, load_data.target)
preprocessor = preprocess.create_preprocessor(imputer_stategy=None, scale_std=False, scale_minmax=False)
        
# Features and target split
features = load_data.features
target = load_data.target

X_train = data[features] # the train data is converted to a numpy array after the fit_transform of the preprocessor
y_train = data[target].to_numpy() # Convert to numpy array

# Preprocess the data
X_train = preprocessor.fit_transform(X_train)        

# Create sequences 
#X_train, y_train = self.create_sequences(X_train, y_train, self.params['seq_length'])

# Create and train the model
lstm_model = LSTMModel(**params)
model_name = lstm_model.model.model_name
lstm_model.fit(X_train, y_train)

# Test the model
data_test = load_data.last_month_data

X_test = data_test[features]
y_test = data_test[target].to_numpy()

# preprocess the test data
X_test = preprocessor.transform(X_test)

# predict
y_pred = lstm_model.predict(X_test)

# Calculate the metrics
model_utils = Model_utils()
mae, mse, rmse, r2 = model_utils.calculate_metrics(y_test, y_pred)
plot_path = model_utils.plot_predictions(y_pred, y_test, mae, mse, rmse, r2, model_name)
print(f'MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}')



## Tracking experiments with MLflow

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
    mlflow.log_params(params)

    # Save the prediction plot
    mlflow.log_artifact(plot_path)
    
    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", comments)

    # Infer the model signature
    signature = infer_signature(X_train.detach().numpy(),
                                    lstm_model.predict(X_train))

    mlflow.pytorch.log_model(lstm_model, model_name,
                            registered_model_name=model_name,
                            signature=signature)
    




