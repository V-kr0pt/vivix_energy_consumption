import mlflow
from mlflow.models import infer_signature

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler#, StandardScaler

from utils.preprocess import Preprocess
from utils.load_data import LoadData 
from utils.model_utils import Model_utils 

from LSTM_model import LSTMModelWrapper as LSTMModel


comments = 'scale_minmax X and y; batch_size=32; lr_schedule after 20 epochs 10percent reduction; target lag; grid search'


# Load data
load_data = LoadData()
data = load_data.data        
# Create the preprocess object and the preprocessor
preprocess = Preprocess(data, load_data.numerical_features, load_data.categorical_features,
                        load_data.boolean_features, load_data.target)
x_preprocessor = preprocess.create_preprocessor(imputer_stategy=None, scale_std=False, scale_minmax=True)


# lagging columns
lag_columns_list = ['medio_diario']*7
lag_values = [1, 2, 3, 4, 5, 6, 7]

# create the lagged columns in data
data = preprocess.create_lag_columns(lag_columns_list, lag_values)
data = data.iloc[7:]

# Features and target split
features = preprocess.features
target = preprocess.target

X_train = data[features] # the train data is converted to a numpy array after the fit_transform of the preprocessor
y_train = data[target].to_numpy() # Convert to numpy array

# Preprocess the data
X_train = x_preprocessor.fit_transform(X_train)  
y_scaler = MinMaxScaler()
y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()  

# Create sequences 
#X_train, y_train = self.create_sequences(X_train, y_train, self.params['seq_length'])

# Create and train the model
#params = {'input_size': X_train.shape[1],
#          'hidden_layer_size': 100,
#          'output_size': 1,
#          'num_layers': 1,
#          'learning_rate': 0.001,
#          'epochs': 10}

param_grid = {'hidden_layer_size': [1000, 1500, 2000, 2500, 3000, 5000],
               'num_layers': [30, 50, 60, 80, 100],
               'learning_rate': [0.0001, 0.0005, 0.001, 0.01, 0.1],
               'epochs': [10, 20, 30, 40, 50]}


#lstm_model = LSTMModel(**params)
#lstm_model.fit(X_train, y_train)

lstm_model = LSTMModel(input_size=X_train.shape[1])
model_name = lstm_model.model.model_name

# grid search
tscv = TimeSeriesSplit(n_splits=10)
grid_search = GridSearchCV(estimator=lstm_model, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error',
                            n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Obtain the best parameters
params = grid_search.best_params_
lstm_model = lstm_model.set_params(**params)
lstm_model.fit(X_train, y_train)

# Test the model
test_data = load_data.last_month_data

# create the lagged columns in data
test_data = preprocess.create_lag_columns(lag_columns_list, lag_values, data=test_data)
test_data = test_data.iloc[7:]

X_test = test_data[features]
y_test = test_data[target].to_numpy()

# preprocess the test data
X_test = x_preprocessor.transform(X_test)

# predict
y_pred = lstm_model.predict(X_test)
y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# Calculate the metrics
model_utils = Model_utils()
mae, mse, rmse, r2 = model_utils.calculate_metrics(y_test, y_pred)
pred_plot_path = model_utils.plot_predictions(y_pred, y_test, mae, mse, rmse, r2, model_name)
print(f'MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}')

# Plotting the Error by epoch
error_plot_path = model_utils.plot_error_by_epoch(lstm_model.model.train_losses, model_name)

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

    # Save the prediction and the error by epoch plot
    mlflow.log_artifact(pred_plot_path)
    mlflow.log_artifact(error_plot_path)
    
    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", comments)

    # Infer the model signature
    signature = infer_signature(X_train, lstm_model.predict(X_train))

    mlflow.pytorch.log_model(lstm_model.model, model_name,
                            registered_model_name=model_name,
                            signature=signature)
    




