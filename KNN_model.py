import mlflow
from mlflow.models import infer_signature
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.neighbors import KNeighborsRegressor as knr
from utils.model_utils import Model_utils 
from utils.load_data import LoadData 
from utils.preprocess import Preprocess

# comments to be saved in the history
comments = '==Sem prod_L, prod_E!===; scale_std = False, scale_minmax = True; Shuffle True'
model_name = 'KNN_3'

# load train/validation data
load_data = LoadData()
data = load_data.data

# Removing prod_l and prod_e columns
data.drop(columns=['prod_l', 'prod_e'], inplace=True)
load_data.boolean_features =[]

# Create the preprocess object and the preprocessor
preprocess = Preprocess(data, load_data.numerical_features, load_data.categorical_features,
                         load_data.boolean_features, load_data.target)
preprocess.create_preprocessor(scale_std=False, scale_minmax=True)

features = preprocess.features
target = preprocess.target

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)


# Preprocess the data
X_train = preprocess.fit_transform(X_train)

# saving the preprocessor
preprocess_path = './results/preprocessors/' + model_name + '_preprocessor.pkl'
preprocess.save_preprocessor(preprocess_path)

# Train the model
# Define the parameter grid for grid search
param_grid = {
    'algorithm': ['auto'],
    'n_neighbors': [4, 5, 6, 7],
    'weights': ['distance', 'uniform'],
    'leaf_size': [1, 10, 15, 20, 25, 30],
    'p': [1, 2]
}

# Define the parameters
#params = {
#    'algorithm':'auto',
#    'leaf_size':1,
#    'n_neighbors':5,
#    'p':2,
#    'weights':'distance'
#}

# Create the KN-Regressor model
model = knr() #knr(**params)

# Train the model
# TimeSeriesSplit Config
tscv = TimeSeriesSplit(n_splits=10)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Set the best parameters 
params = grid_search.best_params_
print(params)
model = model.set_params(**params)
model.fit(X_train, y_train) # train the best model

### Evaluate the model
# Preprocess the test data
X_test = preprocess.transform(X_test) 

# Test the model
y_pred = model.predict(X_test)

# Evaluate the model
model_utils = Model_utils()
mae, mse, rmse, r2 = model_utils.calculate_metrics(y_test, y_pred)
pred_plot_path = model_utils.plot_predictions(y_pred, y_test, mae, mse, rmse, r2, model_name, graph_title='Média de consumo de energia diária do Forno')


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