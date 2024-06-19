import xgboost as xgb
from sklearn.model_selection import train_test_split
from utils.model_utils import Model_utils 
from utils.preprocess import LoadData 

# comments to be saved in the history
comments = 'best model'

load_data = LoadData()

# lagging columns
lag_columns_list = ['medio_diario']*7
lag_values = [1, 2, 3, 4, 5, 6, 7]


data = load_data.create_lag_columns(lag_columns_list, lag_values)
data = data.iloc[7:]

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

# Train the model
model_name = 'xgboost'
# Define the parameter grid for grid search
#param_grid = {
#    'n_estimators': [1200, 1300, 1400],
#    'max_depth': [2, 3, 4],
#    'learning_rate': [0.01, 0.001],
#    'gamma': [0], # Minimum loss reduction required to make a further partition on a leaf node of the tree
#    'subsample': [0.3, 0.5],
#    'reg_alpha': [0.5, 0.6, 0.7], # L1 regularization
#    'reg_lambda': [0], # L2 regularization
#    'random_state': [42]
#}

# Create the XGBRegressor model
model = xgb.XGBRegressor(objective='reg:squarederror', enable_categorical='True',
                         n_estimators= 1300, max_depth= 3, learning_rate= 0.01, 
                         gamma= 0, subsample= 0.3, reg_alpha= 0.5, 
                         reg_lambda= 0, random_state= 42, device='cuda'
                         )
#model = xgb.XGBRegressor(objective='reg:squarederror', enable_categorical='True')
model_utils = Model_utils()

# Train the model with the best parameters
#model_utils.train_model(model, X_train, y_train, model_name, preprocessor=preprocessor, grid_search=True, param_grid=param_grid, comments=comments)
model_utils.train_model(model, X_train, y_train, model_name, preprocessor=preprocessor, grid_search=False, comments=comments)
# Load the model with the best parameters + the preprocessor
model, preprocessor = model_utils.load_model()

# Preprocess the test data
X_test = preprocessor.transform(X_test) 

# Test the model
y_pred = model_utils.test_model(X_test, y_test)