import xgboost as xgb
from sklearn.model_selection import train_test_split
from utils.model_utils import Model_utils 
from utils.preprocess import LoadData 

load_data = LoadData()
data = load_data.data

features = load_data.features
target = load_data.target


X = data[features]
y = data[target]

# Scale is not needed for XGBoost (it is a tree-based model)
preprocessor = load_data.create_preprocessor(scale_std=False, scale_minmax=False)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Train the model
model_name = 'xgboost'
# Define the parameter grid for grid search
param_grid = {
    'n_estimators': [500, 700, 900, 1000],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001],
    'gamma': [0], # Minimum loss reduction required to make a further partition on a leaf node of the tree
    'subsample': [0.1, 0.3, 0.5],
    'reg_alpha': [0.5, 0.7], # L1 regularization
    'reg_lambda': [0], # L2 regularization
    'random_state': [42]
}


# Create the XGBRegressor model
model = xgb.XGBRegressor(objective='reg:squarederror', device='cuda')
model_utils = Model_utils()

# Train the model with the best parameters
model_utils.train_model(model, X_train, y_train, model_name, grid_search=True, param_grid=param_grid)

# Load the model with the best parameters
model = model_utils.load_model()

# Test the model
y_pred = model_utils.test_model(X_test, y_test)