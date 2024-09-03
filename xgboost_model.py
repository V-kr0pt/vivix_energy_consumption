import mlflow
from mlflow.models import infer_signature

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from utils.preprocess import Preprocess
from utils.load_data import LoadData
from utils.model_utils import Model_utils 

import xgboost as xgb

# comments to be saved in the history
comments = '==Sem prod_L, prod_E!===; 7 lag target'
model_name = 'xgboost'

# Load data
load_data = LoadData()
data = load_data.data        

# Removing prod_l and prod_e columns
data.drop(columns=['prod_l', 'prod_e'], inplace=True)
load_data.boolean_features =[]

# Create the preprocess object and the preprocessor
preprocess = Preprocess(data, load_data.numerical_features, load_data.categorical_features,
                        load_data.boolean_features, load_data.target)
preprocess.create_preprocessor(imputer_stategy=None, scale_std=False, scale_minmax=False)

# lagging columns
lag_columns_list = [load_data.target]*7  # load_data.target is the 'consumo_max_diario' column 
lag_values = [1, 2, 3, 4, 5, 6, 7]

# create the lagged columns in data
data = preprocess.create_lag_columns(lag_columns_list, lag_values)
data = data.iloc[7:]

features = preprocess.features
target = preprocess.target

X_train = data[features]
y_train = data[target]

# Preprocess the data
X_train = preprocess.fit_transform(X_train)

# saving the preprocessor
preprocess_path = './results/preprocessors/' + model_name + '_preprocessor.pkl'
preprocess.save_preprocessor(preprocess_path)

# Train the model
# Define the parameter grid for grid search
# param_grid = {...}

params = {
    'n_estimators': 1500,
    'max_depth': 3,
    'learning_rate': 0.5,
    'gamma': 0, # Minimum loss reduction required to make a further partition on a leaf node of the tree
    'subsample': 0.4,
    'reg_alpha': 0.1, # L1 regularization
    'reg_lambda': 0, # L2 regularization
    'random_state': 42
}

# Create the XGBRegressor model
model = xgb.XGBRegressor()

# Train the model
# TimeSeriesSplit Config
#tscv = TimeSeriesSplit(n_splits=10) #SEE THE CHANGES
#grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
#grid_search.fit(X_train, y_train)

# Set the best parameters 
#params = grid_search.best_params_
model = model.set_params(**params)
model.fit(X_train, y_train)

### Evaluate the model
test_data = load_data.last_month_data

# create the lagged columns in data
test_data = preprocess.create_lag_columns(lag_columns_list, lag_values, data=test_data, update_features=False)
test_data = test_data.iloc[7:]

X_test = test_data[features]
y_test = test_data[target]

# Preprocess the test data
X_test = preprocess.transform(X_test) 

# Test the model
y_pred = model.predict(X_test)

# Evaluate the model
model_utils = Model_utils()
mae, mse, rmse, r2 = model_utils.calculate_metrics(y_test, y_pred)
pred_plot_path = model_utils.plot_predictions(y_pred, y_test, mae, mse, rmse, r2, model_name, graph_title='Média de consumo de energia diária do Forno')

# saving the feature importance
all_features = preprocess.features
feature_importance_path = model_utils.plot_feature_importance(model, all_features, model_name)

## --------------------- Tracking experiments with MLflow --------------

#### MLflow

# Set our tracking server uri for logging
dags_hub_url = 'https://dagshub.com/V-kr0pt/vivix_energy_consumption.mlflow'
mlflow.set_tracking_uri(uri=dags_hub_url)

# Create a new MLflow Experiment
experiment = 'energy_regression'
mlflow.set_experiment(experiment)

with mlflow.start_run():
    # Log the loss metricFailed to fetch
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    # Log the hyperparameters
    mlflow.log_params(params)

    # Save the prediction plot
    mlflow.log_artifact(pred_plot_path)
    mlflow.log_artifact(feature_importance_path)
    mlflow.log_artifact(preprocess_path)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", comments)

    # Infer the model signature
    signature = infer_signature(X_train, model.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=model_name,
        signature=signature,
        input_example=X_train[:1], # we can use the first row as an example
        registered_model_name=model_name,
    )