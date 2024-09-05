import mlflow
from mlflow.models import infer_signature

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split

from utils.preprocess import Preprocess
from utils.load_data import LoadData
from utils.model_utils import Model_utils 

import xgboost as xgb

# comments to be saved in the history
comments = '==Sem prod_L, prod_E!===; sem lag; shuffle=True; grid_search'
model_name = 'xgboost_shuffle_v4'

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

features = preprocess.features
target = preprocess.target

X = data[features]
y = data[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# Preprocess the data
X_train = preprocess.fit_transform(X_train)

# saving the preprocessor
preprocess_path = './results/preprocessors/' + model_name + '_preprocessor.pkl'
preprocess.save_preprocessor(preprocess_path)

# Train the model
# Define the parameter grid for grid search
param_grid = {    
    'n_estimators': [100, 300, 500, 700, 1000, 1200, 1500, 1700, 2000],
    'max_depth': [0, 5, 10, 15],
    'learning_rate': [0.01, 0.1, 0.5],
    'gamma': [0], # Minimum loss reduction required to make a further partition on a leaf node of the tree
    'subsample': [0.2, 0.4, 0.6, 0.8],
    'reg_alpha': [0.01, 0.1, 0.2, 0.3], # L1 regularization
    'reg_lambda': [0, 0.001, 0.0025], # L2 regularization
    'random_state': [42]}

# Create the XGBRegressor model
model = xgb.XGBRegressor()

# Train the model
# TimeSeriesSplit Config
tscv = TimeSeriesSplit(n_splits=10) 
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Set the best parameters 
params = grid_search.best_params_
model = model.set_params(**params)
model.fit(X_train, y_train)

### Evaluate the model

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

# save the model in results
model_utils.save_model(model, model_name)


## --------------------- Tracking experiments with MLflow --------------

#### MLflow

# Set our tracking server uri for logging
dags_hub_url = 'https://dagshub.com/V-kr0pt/vivix_energy_consumption.mlflow'
mlflow.set_tracking_uri(uri=dags_hub_url)

# Create a new MLflow Experiment
experiment = 'energy_regression_sem_energia'
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