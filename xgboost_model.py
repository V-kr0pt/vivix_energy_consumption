import xgboost as xgb
import mlflow
from mlflow.models import infer_signature
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score 
from utils.model_utils import Model_utils 
from utils.load_data import LoadData
from utils.preprocess import Preprocess
from imblearn.over_sampling import SMOTE

# comments to be saved in the history
comments = 'shuffle=True'
model_name = 'xgboost'

load_data = LoadData(verbose=True)

# load train/validation data
data = load_data.data
features = load_data.features
target = load_data.target

preprocess = Preprocess(load_data.numerical_features, load_data.categorical_features,
                         load_data.boolean_features)

# train test split
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale is not needed for XGBoost (it is a tree-based model)
preprocess.create_preprocessor(scale_std=False, scale_minmax=False)

# Preprocess the data
X_train = preprocess.fit_transform(X_train)

# Create SMOTE for imbalanced data
smote = SMOTE(sampling_strategy=0.75, random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
new_data_scale = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print(f'new_data_scale: {new_data_scale}')

# Train the model
# Define the parameter grid for grid search
param_grid = {
    'n_estimators': [20, 50, 100, 300, 500, 1000, 1500],  # Número de árvores, pode aumentar dependendo do problema
    'max_depth': [None, 6, 8, 10],  # Profundidade máxima da árvore, maior valor pode capturar mais detalhes, mas pode aumentar o overfitting
    'learning_rate': [0.001, 0.01],  # Taxa de aprendizado, balanceando entre a velocidade e o risco de overfitting
    'reg_alpha': [0.01, 0.1, 0.5],  # Regularização L1
    'reg_lambda': [0.01, 0.1, 0.5],  # Regularização L2
    'scale_pos_weight': [6, 7, 8],  # Ajuste para classes desbalanceadas
    'random_state': [42]  # Para reprodutibilidade
}

# Create the XGboost classification model
model = xgb.XGBClassifier(objective='binary:logistic')

# Train the model
scv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=scv, scoring='average_precision', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Set the best parameters 
params = grid_search.best_params_
model = model.set_params(**params)
model.fit(X_train, y_train)

model_utils = Model_utils()
## Show train results
#y_pred_train_prob = model.predict_proba(X_train)[:,1]

### Evaluate the model
# Preprocess the test data
X_test = preprocess.transform(X_test) 

# Test the model
y_pred_prob = model.predict_proba(X_test)[:,1] # only the probability of the positive class is necessary
y_pred_binary = model.predict(X_test)
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

# Convert the probabilities to binary classes
# threshold -= 0.1 # forcing the threshold to be 10% lower than the optimal threshold to increase recall
# y_pred_binary = (y_pred_prob > threshold).astype(int)
# y_pred_train_binary = (y_pred_train_prob > threshold).astype(int)

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
    mlflow.log_artifact(feature_importance_path)

    # Log the loss
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("auc_roc", auc_roc)
    mlflow.log_metric("auc_pr", auc_pr)
    mlflow.log_metric("threshold", threshold)

    # Save the preprocessor
    mlflow.log_artifact(preprocess_path)

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