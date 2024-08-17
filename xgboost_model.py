import xgboost as xgb
import mlflow
from mlflow.models import infer_signature
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score 
from utils.model_utils import Model_utils 
from utils.load_data import LoadData
from utils.preprocess import Preprocess
from imblearn.over_sampling import SMOTE

# comments to be saved in the history
comments = 'add SMOTE strategy=0.75 with scale_pos_weight + 5 months test; StratifiedKFold shuffle False; average_precision_score; lag only consumo_max_diario; -0.1 threshold'
model_name = 'xgboost'

load_data = LoadData(test_months=5, verbose=True)

# load train/validation data
data = load_data.data

preprocess = Preprocess(data, load_data.numerical_features, load_data.categorical_features,
                         load_data.boolean_features, load_data.target)

# lagging columns
lag_columns_list = ['consumo_max_diario']*7
lag_values = [1, 2, 3, 4, 5, 6, 7]
#lag_columns_list += load_data.features
#lag_values += [1, 2] * len(load_data.features)

# create the lagged columns in data
data = preprocess.create_lag_columns(lag_columns_list, lag_values)
data = data.iloc[7:]

# shuffling data
#data = data.sample(frac=1, random_state=42).reset_index(drop=True)

features = preprocess.features
target = preprocess.target

X_train = data[features]
y_train = data[target]

# Scale is not needed for XGBoost (it is a tree-based model)
preprocessor = preprocess.create_preprocessor(scale_std=False, scale_minmax=False)

# Preprocess the data
X_train = preprocessor.fit_transform(X_train)

# Create SMOTE for imbalanced data
smote = SMOTE(sampling_strategy=0.75, random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
new_data_scale = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print(f'new_data_scale: {new_data_scale}')

# Train the model
# Define the parameter grid for grid search
param_grid = {
    'n_estimators': [300, 500, 1000, 1500],  # Número de árvores, pode aumentar dependendo do problema
    'max_depth': [6, 8, 10],  # Profundidade máxima da árvore, maior valor pode capturar mais detalhes, mas pode aumentar o overfitting
    'learning_rate': [0.001, 0.01],  # Taxa de aprendizado, balanceando entre a velocidade e o risco de overfitting
    'reg_alpha': [0.01, 0.1, 0.5],  # Regularização L1
    'reg_lambda': [0.01, 0.1, 0.5],  # Regularização L2
    'scale_pos_weight': [6, 7, 8],  # Ajuste para classes desbalanceadas
    'random_state': [42]  # Para reprodutibilidade
}


# Create the XGboost classification model
model = xgb.XGBClassifier(objective='binary:logistic')

# Train the model
scv = StratifiedKFold(n_splits=10, shuffle=False)#, random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=scv, scoring='average_precision', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Set the best parameters 
params = grid_search.best_params_
model = model.set_params(**params)
model.fit(X_train, y_train)

model_utils = Model_utils()
## Show train results
y_pred_train_prob = model.predict_proba(X_train)[:,1]

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
y_pred_prob = model.predict_proba(X_test)[:,1]

# plotting ROC and Precision-Recall curves
plot_path_roc, auc_roc, _ = model_utils.plot_roc(y_test, y_pred_prob, model_name)
plot_path_prec_rec, auc_pr, threshold = model_utils.plot_precision_recall(y_test, y_pred_prob, model_name)

# Convert the probabilities to binary classes
threshold -= 0.1 # forcing the threshold to be 10% lower than the optimal threshold to increase recall
y_pred_binary = (y_pred_prob > threshold).astype(int)
y_pred_train_binary = (y_pred_train_prob > threshold).astype(int)

# Evaluate the model
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 =  f1_score(y_test, y_pred_binary)

# Confusion matrix
plot_path_cfm = model_utils.plot_confusion_matrix(y_test, y_pred_binary, model_name)
plot_path_train_cfm = model_utils.plot_confusion_matrix(y_train, y_pred_train_binary, model_name + '_train')

#### MLflow

# Set our tracking server uri for logging
dags_hub_url = 'https://dagshub.com/V-kr0pt/vivix_energy_consumption.mlflow'
mlflow.set_tracking_uri(uri=dags_hub_url)

# Create a new MLflow Experiment
experiment = 'Demanda Classification'
mlflow.set_experiment(experiment)

with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Save the prediction plot
    mlflow.log_artifact(plot_path_cfm)
    mlflow.log_artifact(plot_path_roc)
    mlflow.log_artifact(plot_path_prec_rec)
    mlflow.log_artifact(plot_path_train_cfm)

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
    signature = infer_signature(X_train, model.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="vivix_model",
        signature=signature,
        registered_model_name=model_name,
    )