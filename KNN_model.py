import mlflow
from mlflow.models import infer_signature
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor as knr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.model_utils import Model_utils 
from utils.load_data import LoadData 
from utils.preprocess import Preprocess

# comments to be saved in the history
comments = '7 lagged media_diario + 1 lagged all features - scale_std = True, scale_minmax = False; Shuffle Train Data = False'
load_data = LoadData()

# load train/validation data
data = load_data.data

preprocess = Preprocess(data, load_data.numerical_features, load_data.categorical_features,
                         load_data.boolean_features, load_data.target)

# lagging columns
lag_columns_list = ['medio_diario']*7
lag_values = [1, 2, 3, 4, 5, 6, 7]
lag_columns_list += preprocess.features
lag_values += [1]*len(preprocess.features)

# create the lagged columns in data
data = preprocess.create_lag_columns(lag_columns_list, lag_values)
data = data.iloc[7:]

# shuffle data
#data = data.sample(frac=1).reset_index(drop=True)

features = preprocess.features
target = preprocess.target

X_train = data[features]
y_train = data[target]

# Scale is not needed for XGBoost (it is a tree-based model)
preprocessor = preprocess.create_preprocessor(scale_std=True, scale_minmax=False)

# Preprocess the data
X_train = preprocessor.fit_transform(X_train)

# Train the model
model_name = 'KNN'
# Define the parameter grid for grid search
#param_grid = {
#    'algorithm': ['auto'],
#    'n_neighbors': [4, 5, 6, 7],
#    'weights': ['distance', 'uniform'],
#    'leaf_size': [1],
#    'p': [1, 2]
#}

# Define the parameters
params = {
    'algorithm':'auto',
    'leaf_size':1,
    'n_neighbors':5,
    'p':2,
    'weights':'distance'
}

# Create the KN-Regressor model
model = knr(**params) #knr(**params)

# Train the model
# TimeSeriesSplit Config
tscv = TimeSeriesSplit(n_splits=10)

#grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
#grid_search.fit(X_train, y_train)

# Set the best parameters 
#params = grid_search.best_params_
#model = model.set_params(**params)

# train the best model
model.fit(X_train, y_train)

### Evaluate the model
test_data = load_data.last_month_data

# create the lagged columns in data
test_data = preprocess.create_lag_columns(lag_columns_list, lag_values, data=test_data)
test_data = test_data.iloc[7:]

X_test = test_data[features]
y_test = test_data[target]

# Preprocess the test data
X_test = preprocessor.transform(X_test) 

# Test the model
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

model_utils = Model_utils()
plot_path = model_utils.plot_predictions(y_pred, y_test, mae, mse, rmse, r2, model_name)

#### MLflow

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Vivix")

with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Save the prediction plot
    mlflow.log_artifact(plot_path)

    # Log the loss metric
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", comments)

    # Infer the model signature
    signature = infer_signature(X_train, model.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="vivix_model",
        signature=signature,
        input_example=X_train[0],
        registered_model_name=model_name,
    )

