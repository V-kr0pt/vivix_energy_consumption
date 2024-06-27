from sklearn.neighbors import KNeighborsRegressor as knr
from sklearn.model_selection import train_test_split
from utils.model_utils import Model_utils 
from utils.preprocess import LoadData 


# comments to be saved in the history
comments = 'best KNN model. Removed last month data before shuffle'
load_data = LoadData()

# load train/validation data
data = load_data.data

# lagging columns
lag_columns_list = ['medio_diario']*7
lag_values = [1, 2, 3, 4, 5, 6, 7]

# create the lagged columns in data
data = load_data.create_lag_columns(data, lag_columns_list, lag_values)
data = data.iloc[7:]

features = load_data.features
target = load_data.target

X = data[features]
y = data[target]

# Scale is not needed for XGBoost (it is a tree-based model)
preprocessor = load_data.create_preprocessor(scale_std=True, scale_minmax=False)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
X_train = preprocessor.fit_transform(X_train)

# Train the model
model_name = 'KNN'
# Define the parameter grid for grid search
#param_grid = {
#    'algorithm': ['auto'],
#    'n_neighbors': [2, 5, 10],
#    'weights': ['distance', 'uniform'],
#    'leaf_size': [1, 2, 3],
#    'p': [1, 2]
#}

# Create the KN-Regressor model
model = knr(algorithm='auto', leaf_size=1, n_neighbors= 5, p=1, weights ='distance')

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

