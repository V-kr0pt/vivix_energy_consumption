from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from utils.model_utils import Model_utils 
from utils.load_data import LoadData 
from utils.preprocess import Preprocess

# comments to be saved in the history
comments = 'Best Random Forest Model'
load_data = LoadData()

# load train/validation data
data = load_data.data

preprocess = Preprocess(data, load_data.numerical_features, load_data.categorical_features,
                         load_data.boolean_features, load_data.target)

# lagging columns
lag_columns_list = ['medio_diario']*7
lag_values = [1, 2, 3, 4, 5, 6, 7]

# create the lagged columns in data
data = preprocess.create_lag_columns(lag_columns_list, lag_values)
data = data.iloc[7:]

features = preprocess.features
target = preprocess.target

X = data[features]
y = data[target]

# Scale is not needed for XGBoost (it is a tree-based model)
preprocessor = preprocess.create_preprocessor(scale_std=False, scale_minmax=False)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Train the model
model_name = 'Random_Forest'
# Define the parameter grid for grid search
#param_grid = {
#    'n_estimators': [50, 100, 200, 300, 400, 600, 800, 1000],
#    'max_depth': [None, 5, 10, 15],
#    'min_samples_split': [2, 5, 10],
#    'min_samples_leaf': [1, 2, 4],
#    'max_features': [1, 'sqrt', 'log2'],
#    'random_state': [42]
#}

# Create the Random Forest model
model = RandomForestRegressor(n_estimators=400, max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', random_state=42)
model_utils = Model_utils()

# Train the model with the best parameters
#model_utils.train_model(model, X_train, y_train, model_name, preprocessor=preprocessor, grid_search=True, param_grid=param_grid, comments=comments)
model_utils.train_model(model, X_train, y_train, model_name, preprocessor=preprocessor, grid_search=False, comments=comments)

# Load the model with the best parameters + the preprocessor
model, preprocessor = model_utils.load_model()

# Preprocess the test data (already preprocessed)
#X_test = preprocessor.transform(X_test) 

# Test the model
y_pred = model_utils.test_model(X_test, y_test)
