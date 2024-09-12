import mlflow
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from utils.model_utils import Model_utils 
from utils.load_data import LoadData 
from utils.preprocess import Preprocess

# comments to be saved in the history
comments = 'shuffle=True'
model_name = 'random_forest'

# load train/validation data
load_data = LoadData(verbose=True)

# load train/validation data
data = load_data.data
features = load_data.features
target = load_data.target

preprocess = Preprocess(load_data.numerical_features, load_data.categorical_features,
                         load_data.boolean_features)

# Train test split
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the preprocessor (tree-based models do not need scaling)
preprocess.create_preprocessor(scale_std=False, scale_minmax=False)

# Preprocess the data
X_train = preprocess.fit_transform(X_train)

# Train the model

# Define the parameter grid for grid search
param_grid = {
   'n_estimators': [10, 50, 100, 300, 400, 500, 600, 700],
   'max_depth': [None, 5, 10, 15],
   'min_samples_split': [2, 5],
   'min_samples_leaf': [1, 2, 4],
   'max_features': [1, 'sqrt', 'log2'],
   'random_state': [42]
}

# Create the Random Forest model
model = RandomForestClassifier()

# TimeSeriesSplit Config
scv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Train the model
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=scv, scoring='average_precision', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Set the best parameters 
params = grid_search.best_params_
model = model.set_params(**params)
model.fit(X_train, y_train)

model_utils = Model_utils()
### Evaluate the model
# Preprocess the test data
X_test = preprocess.transform(X_test) 

# Test the model
y_pred_binary = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:,1] # the votes of each tree in the forest
print("\n====================================")
print(f'true values:\n {y_test.values}')
print("=")
print(f'Prediction:\n {y_pred_binary}')
print("=")
print(f'Probability of demand overflow:\n {y_pred_prob}')
print("====================================\n")


# plotting ROC and Precision-Recall curves
plot_path_roc, auc_roc, _ = model_utils.plot_roc(y_test, y_pred_prob, model_name)
plot_path_prec_rec, auc_pr, threshold = model_utils.plot_precision_recall(y_test, y_pred_prob, model_name)

# Evaluate the model
# Evaluate the model
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 =  f1_score(y_test, y_pred_binary)
print("\n==============================")
print(f'Precision: {precision}, Recall: {recall}, F1: {f1}')
print("==============================\n")

# Confusion matrix
plot_path_cfm = model_utils.plot_confusion_matrix(y_test, y_pred_binary, model_name)
#plot_path_train_cfm = model_utils.plot_confusion_matrix(y_train, y_pred_train_binary, model_name + '_train')

# saving the feature importance
all_features = preprocess.features
feature_importance_path = model_utils.plot_feature_importance(model, all_features, model_name)

# save the model in results
model_name = model_utils.save_model(model, model_name)

# saving the preprocessor
preprocess_path = './results/preprocessors/' + model_name + '_preprocessor.pkl'
preprocess.save_preprocessor(preprocess_path)

#### MLflow

# Set our tracking server uri for logging
dags_hub_url = 'https://dagshub.com/V-kr0pt/vivix_energy_consumption.mlflow'
mlflow.set_tracking_uri(uri=dags_hub_url)

# Create a new MLflow Experiment
experiment = 'demand_classification'
mlflow.set_experiment(experiment)

with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Save the prediction plot
    mlflow.log_artifact(plot_path_cfm)
    mlflow.log_artifact(plot_path_roc)
    mlflow.log_artifact(plot_path_prec_rec)
    #mlflow.log_artifact(plot_path_train_cfm)

    # Save the preprocessor
    mlflow.log_artifact(preprocess_path)

    # Log the loss metricFailed to fetch
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("auc_roc", auc_roc)
    mlflow.log_metric("auc_pr", auc_pr)
    mlflow.log_metric("threshold", threshold)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", comments)

    # Infer the model signature
    signature = infer_signature(X_train, model.predict_proba(X_train)[:,1])

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="vivix_model",
        signature=signature,
        registered_model_name=model_name,
    )